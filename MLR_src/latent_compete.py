import torch
from torch import nn
import torch.nn.functional as F

class ViTSlotCompetition(nn.Module):
    def __init__(self, slot_size, num_iterations, mlp_hidden_size, epsilon=1e-8):
        super().__init__()
        self.num_iterations = num_iterations
        self.slot_size = slot_size
        self.epsilon = epsilon

        self.norm_slots  = nn.LayerNorm(slot_size)
        self.norm_inputs = nn.LayerNorm(slot_size)

        self.project_q = nn.Linear(slot_size, slot_size, bias=False)
        self.project_k = nn.Linear(slot_size, slot_size, bias=False)
        self.project_v = nn.Linear(slot_size, slot_size, bias=False)

        self.gru = nn.GRUCell(slot_size, slot_size)
        self.mlp = nn.Sequential(
            nn.LayerNorm(slot_size),
            nn.Linear(slot_size, mlp_hidden_size),
            nn.ReLU(),
            nn.Linear(mlp_hidden_size, slot_size)
        )

    def forward(self, patch_features):
        """
        patch_features: [B, 2, P, D]
        returns: updated slots [B, 2, D]
        """
        B, S, P, D = patch_features.shape
        assert S == 2  

        # 1) initialize slots by mean‐pooling each branch’s patches
        slots = patch_features.mean(dim=2)        # [B, 2, D]

        # 2) flatten all patches into one set of 2*P elements
        patches = patch_features.reshape(B, S*P, D)  # [B, 2*P, D]

        # 3) run competitive slot‐attention
        for _ in range(self.num_iterations):
            slots_prev = slots
            slots      = self.norm_slots(slots)        # [B, 2, D]

            # queries from slots
            q = self.project_q(slots) * (D ** -0.5)    # [B, 2, D]

            # keys/values from patches
            pn = self.norm_inputs(patches)             # [B, 2P, D]
            k  = self.project_k(pn)                    # [B, 2P, D]
            v  = self.project_v(pn)                    # [B, 2P, D]

            # (2P) patches × 2 slots → attention logits
            attn_logits = torch.einsum('bnd,bsd->bns', k, q)  
            attn        = F.softmax(attn_logits, dim=-1)      # [B, 2P, 2]

            # numeric stability + renormalize across patches per slot
            attn = attn + self.epsilon
            attn = attn / (attn.sum(dim=1, keepdim=True) + self.epsilon)  # [B,2P,2]

            # weighted mean of values → updates for each slot
            updates = torch.einsum('bns,bnd->bsd', attn, v)  # [B,2, D]

            # GRU‐update per slot (flatten B×2 → B×2, update, then reshape back)
            slots = self.gru(
                updates.reshape(-1, D), 
                slots_prev.reshape(-1, D)
            ).reshape(B, S, D)

            # small residual MLP
            slots = slots + self.mlp(slots)

        return slots

def create_patch_masks(input_image, num_patches, mask_type='center'):
    """
    Creates binary masks for either the center or outer patches of the image.
    
    Parameters:
    - input_image: the input image tensor (B, N_patches, patch_dim)
    - num_patches: the total number of patches to divide the image into
    - mask_type: 'center' or 'outer' to select the type of mask
    
    Returns:
    - mask: binary mask of shape (B, N_patches) where N_patches is the number of patches
    """
    B, N_patches, patch_dim = input_image.size()
    
    # Generate mask for center or outer patches
    mask = torch.zeros(B, N_patches, dtype=torch.bool, device=input_image.device)

    if mask_type == 'center':
        # For center, select the inner patches (e.g., the central N patches)
        center_indices = range(N_patches // 4, 3 * N_patches // 4)  # Adjust the range based on the center
        mask[:, center_indices] = 1
    elif mask_type == 'outer':
        # For outer, select the patches on the outer parts of the image
        outer_indices = list(range(N_patches // 4)) + list(range(3 * N_patches // 4, N_patches))
        mask[:, outer_indices] = 1
    else:
        raise ValueError("mask_type should be either 'center' or 'outer'")

    return mask


def update_patch_masks_with_gumbel_softmax(scores, obj_init_mask, bg_init_mask, tau=1.0):
    """
    Updates the patch-level masks using Gumbel-Softmax based on patch-level competition scores.
    
    Parameters:
    - scores: tensor of shape [B, N_patches, 2] containing competition scores for object and background
    - obj_init_mask: initial binary mask for the object patches (B, N_patches)
    - bg_init_mask: initial binary mask for the background patches (B, N_patches)
    - tau: temperature for Gumbel-Softmax
    
    Returns:
    - obj_updated_mask: refined mask for object patches (B, N_patches)
    - bg_updated_mask: refined mask for background patches (B, N_patches)
    """
    # Step 1: Apply Gumbel-Softmax to the competition scores
    # The logits will be in the shape [B, N_patches, 2], where each patch has scores for object and background
    logits = scores  # [B, N_patches, 2]
    
    # Apply Gumbel-Softmax to get the refined masks
    assign = F.gumbel_softmax(logits, tau=tau, hard=True)  # [B, N_patches, 2]
    
    # Step 2: Use the Gumbel-Softmax output to update the binary masks (patch-level)
    obj_updated_mask = assign[:, :, 0]  # Object mask (first channel)
    bg_updated_mask = assign[:, :, 1]  # Background mask (second channel)
    
    return obj_updated_mask, bg_updated_mask


def refine_masks_and_images(input_image, obj_patches, bg_patches, obj_slot, bg_slot, num_patches, tau=1.0):
    """
    Refines the object and background patch-level masks and applies them to the input image (patch embeddings).
    
    Parameters:
    - input_image: the original input image tensor (B, N_patches, patch_dim)  # Patch embeddings
    - obj_patches: patch features from object encoder (B, N_patches, patch_dim)
    - bg_patches: patch features from background encoder (B, N_patches, patch_dim)
    - obj_slot: object slot (B, patch_dim)
    - bg_slot: background slot (B, patch_dim)
    - num_patches: the total number of patches to divide the image into
    - tau: temperature for Gumbel-Softmax
    
    Returns:
    - obj_image_refined: the object-only image (after mask refinement)
    - bg_image_refined: the background-only image (after mask refinement)
    """
    B, N_patches, patch_dim = input_image.size()  # Get image dimensions in terms of patches
    
    # Step 1: Create initial object and background masks for the patches
    obj_init_mask = create_patch_masks(input_image, num_patches, mask_type='center')  # Object mask for center patches
    bg_init_mask = create_patch_masks(input_image, num_patches, mask_type='outer')  # Background mask for outer patches
    
    # Step 2: Compute patch-level scores (object vs background) based on patch embeddings and slots
    obj_scores = (obj_patches @ obj_slot.unsqueeze(2)).squeeze(2)  # [B, N_patches]
    bg_scores = (bg_patches @ bg_slot.unsqueeze(2)).squeeze(2)  # [B, N_patches]
    
    # Step 3: Compute competition scores (object vs background)
    scores = torch.stack([obj_scores, bg_scores], dim=-1)  # [B, N_patches, 2]
    
    # Step 4: Update the initial masks using Gumbel-Softmax
    obj_updated_mask, bg_updated_mask = update_patch_masks_with_gumbel_softmax(scores, obj_init_mask, bg_init_mask, tau)
    
    # Step 5: Apply the refined masks to the input patch embeddings (patch-level operation)
    # Each mask will be applied to the corresponding patch embeddings
    obj_image_refined = input_image * obj_updated_mask.unsqueeze(2)  # Apply object mask to patches (B, N_patches, patch_dim)
    bg_image_refined = input_image * bg_updated_mask.unsqueeze(2)  # Apply background mask to patches (B, N_patches, patch_dim)
    
    return obj_image_refined, bg_image_refined




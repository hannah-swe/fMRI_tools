import numpy as np
import nibabel as nib
import numpy.linalg as npl

# Load fMRIPrep output as reference image; Space: MNI152NLin2009cAsym (2mm)
ref_path = "W:/PPPD/Auswertung_Part2/MRI/ROIs/sub-01_task-rest_run-01_space-MNI152NLin2009cAsym_res-2_boldref.nii.gz"
ref_img = nib.load(ref_path)

# Config name, coordinates and radius here:
seed_name = "Precuneus"
center_mni_mm = (1, -61, 38)
radius_mm = 6

def make_sphere_seed(ref_img, center_mni_mm, radius_mm, dtype=np.uint8):
    """
    Creates a binary spherical seed in the exact voxel grid of ref_img.
    center_mni_mm: (x, y, z) in MNI world coordinates (mm)
    radius_mm: sphere radius in mm
    """
    affine = ref_img.affine
    shape  = ref_img.shape[:3]  # 3D grid
    print("ref shape:", ref_img.shape)
    print("zooms:", ref_img.header.get_zooms()[:3])

    # Voxelsize (mm) from affine of reference image
    voxel_sizes = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))

    # Center in mm -> voxel
    inv_aff = npl.inv(affine)
    center_vox = nib.affines.apply_affine(inv_aff, center_mni_mm)

    # Bounding box (in voxels) for speed
    rad_vox = np.ceil(radius_mm / voxel_sizes).astype(int)
    c = np.round(center_vox).astype(int)

    x0, x1 = max(0, c[0]-rad_vox[0]), min(shape[0], c[0]+rad_vox[0]+1)
    y0, y1 = max(0, c[1]-rad_vox[1]), min(shape[1], c[1]+rad_vox[1]+1)
    z0, z1 = max(0, c[2]-rad_vox[2]), min(shape[2], c[2]+rad_vox[2]+1)

    xs = np.arange(x0, x1)
    ys = np.arange(y0, y1)
    zs = np.arange(z0, z1)
    x, y, z = np.meshgrid(xs, ys, zs, indexing="ij")

    # Voxel into mm for calculation of the distance
    vox = np.stack([x, y, z], axis=-1)
    mm  = nib.affines.apply_affine(affine, vox.reshape(-1, 3)).reshape(vox.shape)

    dist = np.sqrt(((mm - np.array(center_mni_mm)) ** 2).sum(axis=-1))
    mask_crop = dist <= radius_mm

    data = np.zeros(shape, dtype=dtype)
    data[x0:x1, y0:y1, z0:z1] = mask_crop.astype(dtype)

    # Take over header (same sform/qform/zooms)
    out = nib.Nifti1Image(data, affine, ref_img.header)
    out.set_data_dtype(dtype)
    return out


# Get seed
seed_img = make_sphere_seed(
    ref_img,
    center_mni_mm=center_mni_mm,
    radius_mm=radius_mm
)

# Save
out_path = f"W:/Hannah/PPPD/{seed_name}_ROI_2009.nii.gz"
nib.save(seed_img, out_path)
print("saved:", out_path)
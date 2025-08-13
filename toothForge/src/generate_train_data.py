import os, csv, logging
import trimesh
import argparse
import numpy as np
from copy import deepcopy

from pathlib import Path

from SpectralMesh import SpectralMesh
from utils import find_rotation_closed_form_iterative

import numpy as np
import pyvista as pv
import matplotlib.pyplot as plt

from scipy.optimize import linear_sum_assignment

#_K_LIMIT: int = 512
_K_LIMIT: int = 25
_RUN_PREALIGNMENT: bool = True

DEBUG = True
OUT_ROOT = '../hungarian/unaligned1'
os.makedirs(OUT_ROOT, exist_ok=True)


_DEFAULT_MESH_DATASET_PATH: str = '../../SpectralCorticalShape/Surfaces/'
_DEFAULT_SPEC_DATASET_PATH: str = '../data/laplace/k25/'

_PERMUTATIONS = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
_FLIPS = [[1, 1, 1] , [-1, 1, 1], [1, -1, 1], [1, 1, -1],
          [-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]]


def parse_args() -> argparse.Namespace:
    args = argparse.ArgumentParser()
    args.add_argument('--folder-path-in', type=str, default=_DEFAULT_MESH_DATASET_PATH,
                      help='Path to dataset folder. This folder contains files in .stl/.obj format.')
    args.add_argument('--folder-path-out', type=str, default=_DEFAULT_SPEC_DATASET_PATH,
                      help='Path to the folder where outputs should be generated.')

    args.add_argument('--k', type=str, default=_K_LIMIT,
                      help=f'How many eigenvectors to generate. Default is {_K_LIMIT}.')

    args = args.parse_args()

    return args


def get_alignment(target_mesh_path: str,
                  case_id: str,
                  mesh_template_spec: SpectralMesh,
                  args,
                  ):
    print(f'Aligning case {case_id}...')
    mesh_target_spec = SpectralMesh(os.path.join(args.folder_path_in, target_mesh_path), k=args.k)

    c_template = np.asarray(mesh_template_spec.graph.eig_vecs.T @ mesh_template_spec.mesh_tri.vertices)
    U_common = mesh_template_spec.graph.eig_vecs

    alignment_opts = {
        'niter': [50, 50, 20, 10, 10],
        'kr': [
            np.arange(4),
            np.arange(6),
            np.arange(8),
            np.arange(20),
            np.arange(args.k),
        ],
    }
    
    if _RUN_PREALIGNMENT:
        avg_volume_ins = (mesh_template_spec.mesh_tri.volume + mesh_target_spec.mesh_tri.volume) / 2.

        lowest_err = float('inf')
        best_perm = [0, 1, 2]  # Defaults
        best_flip = [1, 1, 1]  # Defaults
        for perm in _PERMUTATIONS:
            for flip in _FLIPS:
                m_g_work_copy = deepcopy(mesh_target_spec)
                m_g_work_copy.graph.X[:, :3] = m_g_work_copy.graph.X[:, perm]
                m_g_work_copy.graph.eig_vecs[:, :3] = m_g_work_copy.graph.eig_vecs[:, perm]
                m_g_work_copy.graph.eig_vecs_inv[:3, :] = m_g_work_copy.graph.eig_vecs_inv[perm, :]

                for eigenmode_id in [0, 1, 2]:
                    m_g_work_copy.graph.X[:, eigenmode_id] *= flip[eigenmode_id]
                    m_g_work_copy.graph.eig_vecs[:, eigenmode_id] *= flip[eigenmode_id]
                    m_g_work_copy.graph.eig_vecs_inv[eigenmode_id, :] *= flip[eigenmode_id]

                corr = find_rotation_closed_form_iterative(mesh_template_spec, m_g_work_copy, alignment_opts)

                w_f = mesh_template_spec.graph.eig_vecs.T @ mesh_template_spec.mesh_tri.vertices  # Shape of w_f: [k, 3]
                w_g = m_g_work_copy.graph.eig_vecs.T @ m_g_work_copy.mesh_tri.vertices  # Shape of w_g: [k, 3]
                U_common = mesh_template_spec.graph.eig_vecs

                try:
                    temp_w_g = corr['R_21'] @ w_g
                except ValueError:
                    continue

                w_p = .5 * w_f + .5 * temp_w_g  # shape of w_p: [k, 3]
                pos_p = U_common @ w_p  # Reconstruction, shape: [N, 3]

                mesh_tri = trimesh.Trimesh(vertices=pos_p, faces=mesh_template_spec.mesh_tri.faces)
                mid_shape_volume = np.abs(mesh_tri.volume)
                diff = np.abs(mid_shape_volume - avg_volume_ins)
                if diff < lowest_err:
                    lowest_err = diff
                    best_perm = perm
                    best_flip = flip
                    print(f'Best-performing combination: permutation: {perm}, flip: {flip}')

        mesh_target_spec.graph.X[:, :3] = mesh_target_spec.graph.X[:, best_perm]
        mesh_target_spec.graph.eig_vecs[:, :3] = mesh_target_spec.graph.eig_vecs[:, best_perm]
        mesh_target_spec.graph.eig_vecs_inv[:3, :] = mesh_target_spec.graph.eig_vecs_inv[best_perm, :]

        for eigenmode_id in [0, 1, 2]:
            mesh_target_spec.graph.X[:, eigenmode_id] *= best_flip[eigenmode_id]
            mesh_target_spec.graph.eig_vecs[:, eigenmode_id] *= best_flip[eigenmode_id]
            mesh_target_spec.graph.eig_vecs_inv[eigenmode_id, :] *= best_flip[eigenmode_id]

    
    corr = find_rotation_closed_form_iterative(mesh_template_spec, mesh_target_spec, alignment_opts)

    # Store reconstructions as well.
        
    c_unaligned = np.asarray(mesh_target_spec.graph.eig_vecs.T @ mesh_target_spec.mesh_tri.vertices)
    c_aligned = corr['R_21'] @ c_unaligned
    c_intp = .5 * c_template + .5 * c_aligned

    pos_intp = U_common @ c_intp
    mesh_tri_intp = trimesh.Trimesh(vertices=pos_intp, faces=mesh_template_spec.mesh_tri.faces)

    pos_unaligned = U_common @ c_unaligned
    mesh_tri_unaligned = trimesh.Trimesh(vertices=pos_unaligned, faces=mesh_template_spec.mesh_tri.faces)

    pos_aligned = U_common @ c_aligned
    mesh_tri_aligned = trimesh.Trimesh(vertices=pos_aligned, faces=mesh_template_spec.mesh_tri.faces)

    print(f'Case {case_id} successfully aligned.')

    return mesh_target_spec, corr, mesh_tri_intp, mesh_tri_unaligned, mesh_tri_aligned, case_id


def main() -> None:
    args = parse_args()

    os.makedirs(args.folder_path_out, exist_ok=True)  # Create empty output folder if it does not exist yet.

    mesh_ps = [f for f in os.listdir(args.folder_path_in) if f.endswith('.surf.gii')]

    ''' Processing template mesh. '''
    mesh_template_p = mesh_ps[0]  # First mesh will be set as the template.
    mesh_template_spec = SpectralMesh(os.path.join(args.folder_path_in, mesh_template_p), k=args.k)

    template_f_p = os.path.join(args.folder_path_out, 'template-sub-001_species-Cercopithecus+cephus_hemi-L')
    os.makedirs(template_f_p, exist_ok=True)
    mesh_template_spec.store_data(template_f_p, is_template=True)

    c_template = np.asarray(mesh_template_spec.graph.eig_vecs.T @ mesh_template_spec.mesh_tri.vertices)
    pos_template = mesh_template_spec.graph.eig_vecs @ c_template
    mesh_tri_template = trimesh.Trimesh(vertices=pos_template, faces=mesh_template_spec.mesh_tri.faces)
    mesh_tri_template.export(os.path.join(args.folder_path_out, 'template-sub-001_species-Cercopithecus+cephus_hemi-L', 'reconstructed_aligned.stl'))

    target_meshes_paths = mesh_ps[1:]
    #target_ids = [str(i).zfill(6) for i in range(len(target_meshes_paths))]
    target_ids = [Path(path).stem.split('.')[0] for path in target_meshes_paths]

    results = []
    off_cases = [       
                "sub-018_species-Gorilla+beringei_hemi-L", "sub-026_species-Lophocebus+albigena_hemi-L", "sub-032_species-Trachypithecus+germaini_hemi-L",
                "sub-042_species-Microtus+ochrogaster_hemi-L", "sub-050_species-Galeopterus+variegatus_hemi-L", "sub-053_species-Euchoreutes+naso_hemi-L",
                "sub-055_species-Ochotona+macrotis_hemi-L", "sub-057_species-Perodicticus+potto_hemi-L", "sub-062_species-Macaca+fuscata_hemi-L",
                "sub-073_species-Cercopithecus+nictitans_hemi-L", "sub-074_species-Colobus+guereza_hemi-L", "sub-076_species-Galago+senegalensis_hemi-L",
                "sub-077_species-Leontopithecus+rosalia_hemi-L"
    ]
    ''' Processing other samples. '''
    for sample, target_id in zip(target_meshes_paths, target_ids):
        #if target_id not in off_cases:
         #   continue
        try:
            
            case_dir = os.path.join(OUT_ROOT, f"case_{target_id}")
            os.makedirs(case_dir, exist_ok=True)
                
            mesh_target_spec, corr, mesh_tri_intp, mesh_tri_unaligned, mesh_tri_aligned, _ = get_alignment(sample,target_id,mesh_template_spec,args)
                
            mesh_target_spec.store_data(os.path.join(args.folder_path_out, target_id), R=corr['R_21'])
        
            mesh_tri_intp.export(os.path.join(args.folder_path_out, target_id, 'reconstructed_interp.stl'))
            mesh_tri_unaligned.export(os.path.join(args.folder_path_out, target_id, 'reconstructed_unaligned.stl'))             
            mesh_tri_aligned.export(os.path.join(args.folder_path_out, target_id, 'reconstructed_aligned.stl'))

        except:
            print(f"Hey I couldnt with file {target_id}")
       
        if DEBUG:
            diff = mesh_tri_aligned.vertices - mesh_template_spec.coords
            per_vtx_err = np.linalg.norm(diff, axis=1)
            mse_final = float(np.mean(per_vtx_err**2))
            pct_bad = float((per_vtx_err > 1e-3).mean())
            vol_err = float(abs(mesh_tri_aligned.volume - mesh_template_spec.mesh_tri.volume))
    
            results.append({
                "case_id":target_id,
                "mse_final":mse_final,
                "pct_bad_vert": pct_bad,
                "volume_err": vol_err,
                "cond_R21": float(np.linalg.cond(corr["R_21"]))
            })
    
        
            import matplotlib.pyplot as plt
            
            # per vertex error as colored mesh
            # save the per-vertex error as a colored mesh
            
            p = pv.Plotter(off_screen=True)
            p.add_mesh(mesh_tri_aligned, scalars=per_vtx_err, cmap="hot", clim=[0, per_vtx_err.max()])

            p.camera_position = [
                            (-1, 0, 0), 
                            (0, 0, 0),   
                            (0, 0, 1)    
            ]
            p.reset_camera()

            p.screenshot(os.path.join(case_dir, "aligned_error.png"))
            p.close()
        
            # save the err_X, err_Z curves

            plt.figure()
            plt.plot(corr["errs_X"], label="err_X (modes 0-2)"); plt.plot(corr["errs_Z"], label="err_Z  (all features)")
            plt.legend(); plt.title(f"case {target_id} – spectral ICP errors")
            plt.savefig(os.path.join(case_dir, "icp_errors.png"))
            plt.close()
                  
            # save the last corr12 scatter
            plt.figure()
            plt.scatter(np.arange(len(corr["corr_12"])), corr["corr_12"], s=1)
            plt.title(f"case {target_id} – corr12 final")
            plt.savefig(os.path.join(case_dir, "corr12.png"))
            plt.close()

            


if __name__ == '__main__':
    main()

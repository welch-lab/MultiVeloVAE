import os
import sys
import pytest
import numpy as np
import scanpy as sc
import torch

current_path = os.path.dirname(__file__)
src_path = os.path.join(current_path, "../")
sys.path.append(src_path)


@pytest.fixture(scope="session")
def input_adata():
    in_dat = sc.read_h5ad('test_files/test_input.h5ad')
    return in_dat


def test_init_params(input_adata):
    from multivelovae.model import init_params

    c = input_adata.layers['Mc']
    u = input_adata.layers['Mu']
    s = input_adata.layers['Ms']

    out = init_params(c[:,:5], u[:,:5], s[:,:5],
                      98,
                      fit_scaling=True,
                      global_std=True,
                      tmax=1,
                      rna_only=False,
                      rna_only_idx=None)

    assert np.allclose(out[0], np.array([5.26051238,3.50297445,5.25575955,5.90515433,4.27038595]), rtol=1e-3, atol=1e-6)
    assert np.allclose(out[1], np.array([24.18441047,2.25580724,3.9283581,14.91288134,1.48065427]), rtol=1e-3, atol=1e-6)
    assert np.allclose(out[2], np.array([12.17217217,34.27427427,8.78878879,9.94994995,9.98998999]), rtol=1e-3, atol=1e-6)
    assert np.allclose(out[3], np.array([6.18889512,3.66308031,5.90132336,6.00002771,4.42303168]), rtol=1e-3, atol=1e-6)
    assert np.allclose(out[4], np.array([1.1012043,1.04944689,0.92160792,0.91724866,1.01929808]), rtol=1e-3, atol=1e-6)


def test_VAEChrom(input_adata):
    from multivelovae.model import VAEChrom, differential_dynamics
    torch.manual_seed(2022)
    np.random.seed(2022)

    # check initialization
    model = VAEChrom(input_adata,
                     input_adata,
                     device='cuda:0',
                     cluster_key="leiden",
                     embed="umap")
    assert np.allclose(input_adata.var['decoder_init_alpha_c'].values[:10], model.decoder.alpha_c.detach().cpu().numpy()[:10], rtol=1e-3, atol=1e-6)
    assert np.allclose(input_adata.var['decoder_init_alpha'].values[:10], model.decoder.alpha.detach().cpu().numpy()[:10], rtol=1e-3, atol=1e-6)
    assert np.allclose(input_adata.var['decoder_init_beta'].values[:10], model.decoder.beta.detach().cpu().numpy()[:10], rtol=1e-3, atol=1e-6)
    assert np.allclose(input_adata.var['decoder_init_gamma'].values[:10], model.decoder.gamma.detach().cpu().numpy()[:10], rtol=1e-3, atol=1e-6)
    assert np.allclose(input_adata.var['decoder_init_scaling_c'].values[:10], model.decoder.scaling_c.detach().cpu().numpy()[:10], rtol=1e-3, atol=1e-6)
    assert np.allclose(input_adata.var['decoder_init_scaling_u'].values[:10], model.decoder.scaling_u.detach().cpu().numpy()[:10], rtol=1e-3, atol=1e-6)
    assert np.allclose(input_adata.var['decoder_init_scaling_s'].values[:10], model.decoder.scaling_s.detach().cpu().numpy()[:10], rtol=1e-3, atol=1e-6)

    # check training
    model.train()
    assert np.allclose(input_adata.var['decoder_post_alpha_c'].values[:10], model.decoder.alpha_c.detach().cpu().numpy()[:10], rtol=1e-3, atol=1e-6)
    assert np.allclose(input_adata.var['decoder_post_alpha'].values[:10], model.decoder.alpha.detach().cpu().numpy()[:10], rtol=1e-3, atol=1e-6)
    assert np.allclose(input_adata.var['decoder_post_beta'].values[:10], model.decoder.beta.detach().cpu().numpy()[:10], rtol=1e-3, atol=1e-6)
    assert np.allclose(input_adata.var['decoder_post_gamma'].values[:10], model.decoder.gamma.detach().cpu().numpy()[:10], rtol=1e-3, atol=1e-6)
    assert np.allclose(input_adata.var['decoder_post_scaling_c'].values[:10], model.decoder.scaling_c.detach().cpu().numpy()[:10], rtol=1e-3, atol=1e-6)
    assert np.allclose(input_adata.var['decoder_post_scaling_u'].values[:10], model.decoder.scaling_u.detach().cpu().numpy()[:10], rtol=1e-3, atol=1e-6)
    assert np.allclose(input_adata.var['decoder_post_scaling_s'].values[:10], model.decoder.scaling_s.detach().cpu().numpy()[:10], rtol=1e-3, atol=1e-6)

    # check likelihood calculation and velocity prediction
    model.save_anndata()
    assert np.allclose(input_adata.var['post_likelihood'].values[:10], input_adata.var['vae_likelihood'][:10], rtol=1e-3, atol=1e-6)
    assert np.array_equal(input_adata.var['post_velocity_genes'].values, input_adata.var['vae_velocity_genes'].values)

    # check differential test
    a = np.isin(input_adata.obs['leiden'], ['V-SVZ'])
    b = np.isin(input_adata.obs['leiden'], ['nIPC', 'Upper Layer'])
    df_dd = differential_dynamics(input_adata,
                                  input_adata,
                                  model=model,
                                  idx1=a, idx2=b,
                                  mode='change',
                                  test_decoupling=True)
    assert np.allclose(input_adata.var['log2_diff_decoupling'].values, df_dd.loc[input_adata.var_names, 'log2_diff_decoupling'].values, rtol=1e-3, atol=1e-6)

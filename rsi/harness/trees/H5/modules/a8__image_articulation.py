"""Bounded anatomical image-supported articulation residual within one segment.

Original PyTorch/NumPy implementation using supplied MANO. Image translation
and scale are nuisance variables; a strong angular prior retains the current
kinematics. The supplied MANO pose basis restricts corrections to common grasp
directions; a Cauchy residual rejects inconsistent finger anchors. Only visible
observations contribute. MANO: Romero et al., ACM TOG 2017; original BSD-3-Clause
wrapper with the supplied scientific-research MANO model, no added weights.
"""
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.ndimage import gaussian_filter1d
_TIPS=[744,320,443,554,671]
_ORDER=[0,13,14,15,16,1,2,3,17,4,5,6,18,10,11,12,19,7,8,9,20]

from functools import lru_cache
@lru_cache(maxsize=2)
def _basis(side):
 import smplx
 from ace_eval.mano import MANO_DIR
 model=smplx.create(MANO_DIR, model_type='mano', is_rhand=bool(side), use_pca=True, num_pca_comps=44, flat_hand_mean=False)
 basis=model.hand_components.detach().numpy()
 basis=basis/np.maximum(np.linalg.norm(basis,axis=1,keepdims=True),1e-9)
 return basis

def apply(state,seg,params):
 if 'exists_2d' not in seg.raw:return state
 import torch
 from harness.core import mano_models
 torch.set_num_threads(1)
 n=seg.n
 base_geometry = seg.mano(dict(state))
 relative_old = state['joints_cam'] - state['joints_cam'][:, :, :1]
 relative_base = base_geometry['joints_cam'] - base_geometry['joints_cam'][:, :, :1]
 mesh_scale = np.sum(relative_old * relative_base, axis=(2, 3)) / np.maximum(np.sum(relative_base ** 2, axis=(2, 3)), 1e-12)
 mesh_scale = np.clip(np.nan_to_num(mesh_scale, nan=1.0), 0.8, 1.2)
 old={k:state[k].copy() for k in ('hp','transl','joints_cam','verts_cam')}
 vis=np.clip(np.asarray(seg.raw['exists_2d'][:n],float),0,1)
 target=np.asarray(state['joints_2d'],float)
 if params.get('raw_2d',False): target=np.asarray(seg.raw['direct_joints2d'][:n])*[seg.W,seg.H]
 K=torch.tensor(seg.K,dtype=torch.float32)
 maximum=np.deg2rad(float(params.get('max_degrees',5)))
 changed=False
 for side,model in mano_models().items():
  finite_hp=np.isfinite(state['hp'][:,side]).all((1,2,3))
  good=(vis[:,side]>.8)&finite_hp&np.isfinite(state['go'][:,side]).all((1,2))&np.isfinite(state['betas'][:,side]).all(1)&np.isfinite(target[:,side]).all((1,2))&(state['joints_cam'][:,side,:,2]>.05).all(1)
  idx=np.flatnonzero(good)
  if len(idx)<3: continue
  idx=idx[np.unique(np.linspace(0,len(idx)-1,min(len(idx),int(params.get('samples',12)))).astype(int))]
  safe_hp=np.where(finite_hp[:,None,None,None],state['hp'][:,side],np.eye(3))
  aa=Rotation.from_matrix(safe_hp.reshape(-1,3,3)).as_rotvec().reshape(n,45)
  h0=torch.tensor(aa[idx],dtype=torch.float32)
  go=torch.tensor(Rotation.from_matrix(state['go'][idx,side]).as_rotvec(),dtype=torch.float32)
  beta=torch.tensor(state['betas'][idx,side],dtype=torch.float32)
  wrist=torch.tensor(state['joints_cam'][idx,side,0],dtype=torch.float32)
  size=torch.tensor(mesh_scale[idx,side,None,None],dtype=torch.float32)
  tar=torch.tensor(target[idx,side],dtype=torch.float32)
  tar=tar-tar.mean(1,keepdim=True)
  extent=tar.square().sum(-1).mean(-1).sqrt().clamp(min=10)
  tar=tar/extent[:,None,None]
  shared=params.get('shared',True)
  basis=torch.tensor(_basis(side)[:int(params.get('components',30))],dtype=torch.float32) if int(params.get('components',45))<45 else torch.eye(45)
  delta=torch.zeros((1 if shared else len(idx),len(basis)),requires_grad=True)
  for p in model.parameters():p.requires_grad_(False)
  optimizer=torch.optim.LBFGS([delta],lr=1,max_iter=int(params.get('steps',12)),max_eval=20,history_size=8,tolerance_grad=1e-8,tolerance_change=1e-10,line_search_fn='strong_wolfe')
  def closure():
   optimizer.zero_grad()
   out=model(global_orient=go,hand_pose=h0+delta@basis,betas=beta)
   joints=torch.cat([out.joints,out.vertices[:,_TIPS]],1)[:,_ORDER]
   xyz=size*(joints-joints[:,:1])+wrist[:,None]
   z=xyz[:,:,2].clamp(min=.03)
   uv=torch.stack([K[0,0]*xyz[:,:,0]/z+K[0,2],K[1,1]*xyz[:,:,1]/z+K[1,2]],-1)
   uv=uv-uv.mean(1,keepdim=True)
   uv=uv/uv.square().sum(-1).mean(-1).sqrt().clamp(min=10)[:,None,None]
   residual=uv-tar
   q=residual.square().sum(-1)
   scale=float(params.get('robust_scale',1e6))
   data=(scale**2*torch.log1p(q/scale**2)).sum(-1).mean()
   loss=data+float(params.get('prior',1))*delta.square().sum(-1).mean()
   loss.backward()
   return loss
  with torch.enable_grad():optimizer.step(closure)
  d=(delta@basis).detach().numpy().reshape(-1,15,3)
  if not np.isfinite(d).all():continue
  d*=np.minimum(1,maximum/np.maximum(np.linalg.norm(d,axis=-1),1e-9))[...,None]
  if shared:d=np.broadcast_to(d,(n,15,3)).copy()
  else:
   d=np.stack([np.interp(np.arange(n),idx,d[:,k,c]) for k in range(15) for c in range(3)],axis=1).reshape(n,15,3)
   if params.get('sigma',2)>0:d=gaussian_filter1d(d,float(params.get('sigma',2)),axis=0)
  carry=float(params.get('carry_frames',15))
  distance=np.abs(np.arange(n)[:,None]-np.flatnonzero(good)).min(1)
  fade=np.exp(-distance/max(carry,1e-6))
  d*=fade[:,None,None]*float(params.get('alpha',1))
  updated=Rotation.from_rotvec((aa.reshape(n,15,3)+d).reshape(-1,3)).as_matrix().reshape(n,15,3,3)
  state['hp'][finite_hp,side]=updated[finite_hp]
  changed=True
 if not changed:return state
 state=seg.mano(state)
 center = state['joints_cam'][:, :, :1].copy()
 state['joints_cam'] = center + mesh_scale[..., None, None] * (state['joints_cam'] - center)
 state['verts_cam'] = center + mesh_scale[..., None, None] * (state['verts_cam'] - center)
 shift=old['joints_cam'][:,:,0]-state['joints_cam'][:,:,0]
 for k in ('transl','joints_cam','verts_cam'):state[k]+=shift if k=='transl' else shift[:,:,None]
 def visible(j):
  uv=seg.project(j)
  return ((j[...,2]>.01)&(uv[...,0]>=0)&(uv[...,0]<seg.W)&(uv[...,1]>=0)&(uv[...,1]<seg.H)).any(-1)
 crossing=visible(old['joints_cam'])!=visible(state['joints_cam'])
 for k in old:state[k][crossing]=old[k][crossing]
 return state

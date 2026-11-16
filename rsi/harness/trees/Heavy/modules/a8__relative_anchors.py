"""Final relative 2D kinematic consensus after image pose fitting.

Independent image evidence retains the common image position, while the
refined MANO projection regularizes the relative joint layout.
Original NumPy implementation (BSD-3-Clause)."""
import numpy as np

def apply(state,seg,params):
 uv=np.asarray(state['joints_2d'],float)
 proj=seg.project(state['joints_cam'])
 d=proj-uv
 if params.get('relative',False):d-=d.mean(2,keepdims=True)
 weight=float(params.get('weight',.25))
 if params.get('edge',False):
  p=seg.project(state['joints_cam'])
  inside=(p[...,0]>=0)&(p[...,0]<seg.W)&(p[...,1]>=0)&(p[...,1]<seg.H)
  n=inside.sum(2)
  weight=weight*((n>0)&(n<21))[...,None,None]
 good=np.isfinite(d).all((2,3))&(state['joints_cam'][...,2]>.05).all(2)
 correction=weight*d
 norm=np.linalg.norm(correction,axis=-1)
 correction*=np.minimum(1.,float(params.get('max_pixels',40.))/np.maximum(norm,1e-9))[...,None]
 state['joints_2d']=uv+np.where(good[:,:,None,None],correction,0)
 return state

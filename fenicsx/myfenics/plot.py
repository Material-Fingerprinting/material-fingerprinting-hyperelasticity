#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 26 09:47:47 2024

@author: mflaschel
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation

# plt.rcParams['text.usetex'] = True

COLOR0 = np.array([255, 255, 255])/255 # white
COLOR1 = np.array([255, 0, 0])/255 # red
COLOR2 = np.array([0, 255, 0])/255 # green
COLOR3 = np.array([0, 0, 255])/255 # blue
COLOR4 = np.array([255, 255, 0])/255 # yellow
COLOR5 = np.array([255, 0, 255])/255 # purple
COLOR6 = np.array([0, 255, 255])/255 # turquise

COLOR7 = np.array([255, 58, 0])/255 # CME red
COLOR8 = np.array([255, 146, 0])/255 # CME orange
COLOR9 = np.array([255, 245, 0])/255 # CME yellow
COLOR10 = np.array([0, 255, 220])/255 # CME turquise

COLOR11 = np.array([255, 153, 204])/255 # pastel purple
COLOR12 = np.array([153, 204, 255])/255 # pastel blue
COLOR13 = np.array([255, 255, 153])/255 # pastel yellow
COLOR14 = np.array([204, 255, 255])/255 # pastel offwhite

COLOR15 = np.array([23, 236, 236])/255 # turquise (Ocean)

def plot_mesh(ax,nodes,elements,linewidth=1,color=COLOR0):
    dim = nodes.shape[1]
    if dim == 2:
        n_nodes_per_element = elements.shape[1]
        if n_nodes_per_element == 3:
            # plot mesh with 3-node triangular elements
            n_elements = elements.shape[0]
            for i in range(n_elements):
                element_nodes = nodes[elements[i],:]
                ax.plot(element_nodes[[0,1],0], element_nodes[[0,1],1], linewidth = linewidth, color=color)
                ax.plot(element_nodes[[0,2],0], element_nodes[[0,2],1], linewidth = linewidth, color=color)
                ax.plot(element_nodes[[1,2],0], element_nodes[[1,2],1], linewidth = linewidth, color=color)
        elif n_nodes_per_element == 4:
            # plot mesh with 4-node quadrilateral elements
            n_elements = elements.shape[0]
            for i in range(n_elements):
                element_nodes = nodes[elements[i],:]
                ax.plot(element_nodes[[0,1],0], element_nodes[[0,1],1], linewidth = linewidth, color=color)
                ax.plot(element_nodes[[0,2],0], element_nodes[[0,2],1], linewidth = linewidth, color=color)
                ax.plot(element_nodes[[1,3],0], element_nodes[[1,3],1], linewidth = linewidth, color=color)
                ax.plot(element_nodes[[2,3],0], element_nodes[[2,3],1], linewidth = linewidth, color=color)
        
    elif dim == 3:
        # plot mesh with 4-node tetrahedral elements
        n_elements = elements.shape[0]
        for i in range(n_elements):
            element_nodes = nodes[elements[i],:]
            ax.plot(element_nodes[[0,1],0], element_nodes[[0,1],1], element_nodes[[0,1],2], linewidth = linewidth, color=color)
            ax.plot(element_nodes[[0,2],0], element_nodes[[0,2],1], element_nodes[[0,2],2], linewidth = linewidth, color=color)
            ax.plot(element_nodes[[0,3],0], element_nodes[[0,3],1], element_nodes[[0,3],2], linewidth = linewidth, color=color)
            ax.plot(element_nodes[[1,2],0], element_nodes[[1,2],1], element_nodes[[1,2],2], linewidth = linewidth, color=color)
            ax.plot(element_nodes[[1,3],0], element_nodes[[1,3],1], element_nodes[[1,3],2], linewidth = linewidth, color=color)
            ax.plot(element_nodes[[2,3],0], element_nodes[[2,3],1], element_nodes[[2,3],2], linewidth = linewidth, color=color)
    return ax

def plot_plane(ax,nodes,elements,alpha=1.0,color=COLOR14):
    # plot a two-dimensional computational domain given the mesh
    n_elements = elements.shape[0]
    for i in range(n_elements):
    # for i in range(1):
        coo = nodes[elements[i,:]] # coordinates of the element
        # increase element slightly
        # factor = 1.1
        # x_min = np.min(coo[:,0])
        # x_max = np.max(coo[:,0])
        # y_min = np.min(coo[:,1])
        # y_max = np.max(coo[:,1])
        # coo[:,0] = (x_max-x_min)/2 + factor * (coo[:,0] - (x_max-x_min)/2)
        # coo[:,1] = (y_max-y_min)/2 + factor * (coo[:,1] - (y_max-y_min)/2)
        ax.add_collection(PolyCollection([coo],alpha=alpha,color=color,linewidths=.1,edgecolor=color))
    return ax

def plot_boundary(ax,nodes,elements,boundaries,alpha=.8,color=COLOR14):
    # plot the boundary of a computational domain given the mesh
    n_elements = elements.shape[0]
    DEBUG = False
    counter = 0
    for i in range(n_elements):
        for b in boundaries:
            isonboundary = b[elements[i]]
            if np.sum(isonboundary) == 3:
                coo = [nodes[elements[i,isonboundary],:]] # coordinates of the element boundary surface
                normal = np.cross(coo[0][0]-coo[0][1],coo[0][0]-coo[0][2])
                normal = np.abs(normal)/np.linalg.norm(normal)
                light = np.array([-0.25,-0.15,0.1])
                light = light/np.linalg.norm(light)
                factor = (np.dot(normal,light) + 1) / 2
                if DEBUG:
                    if counter == 0:
                        print('Add surface!')
                        ax.add_collection(Poly3DCollection(coo,alpha=alpha,color=factor**(1/4)*color,linewidths=.0))
                    counter = counter + 1
                else:
                    ax.add_collection(Poly3DCollection(coo,alpha=alpha,color=factor**(1/4)*color,linewidths=.0))
    return ax

def plot_vector_field(ax,nodes,vector,linewidths=0.3,color=COLOR13):
    # plot vector field
    # TODO: remove arrows if there are too many
    nodes_arrows = nodes
    vector_arrows = vector
    ax.quiver(nodes[:,0], nodes[:,1], nodes[:,2],
              vector[:,0], vector[:,1], vector[:,2],linewidths=linewidths,length=1.0,normalize=False,color=color)
    return ax

def plot_strain_field(ax,nodes,eps_nodes,scale_sphere=0.02,scale_def=1,alpha=.8,color=COLOR5):
    def get_u(eps,x,y,z):
        eps11 = eps[0,0]
        eps22 = eps[1,1]
        eps33 = eps[2,2]
        eps12 = eps[0,1]
        eps13 = eps[0,2]
        eps23 = eps[1,2]
        ux = eps11 * x + eps12 * y + eps13 * z
        uy = eps12 * x + eps22 * y + eps23 * z
        uz = eps13 * x + eps23 * y + eps33 * z
        return ux, uy, uz
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = scale_sphere * np.cos(u) * np.sin(v)
    y = scale_sphere * np.sin(u) * np.sin(v)
    z = scale_sphere * np.cos(v)
    for i in range(nodes.shape[0]):
        node_sphere = nodes[i,:]
        eps_sphere = eps_nodes[i,:,:]
        ux, uy, uz = get_u(eps_sphere,x,y,z)
        x_sphere = x + node_sphere[0]
        y_sphere = y + node_sphere[1]
        z_sphere = z + node_sphere[2]
        x_sphere_def = x_sphere + scale_def*ux
        y_sphere_def = y_sphere + scale_def*uy
        z_sphere_def = z_sphere + scale_def*uz
        ax.plot_surface(x_sphere_def,y_sphere_def,z_sphere_def,alpha=alpha,color=color)
    return ax

def plot_stress_field(ax,nodes,sig_nodes,only_half=None,scale_sphere=0.02,alpha=.4,color=COLOR8):
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = scale_sphere * np.cos(u) * np.sin(v)
    y = scale_sphere * np.sin(u) * np.sin(v)
    z = scale_sphere * np.cos(v)
    normal_position_nomalized = np.array([
        [1,0,0],
        [0,1,0],
        [0,0,1],
        [-1,0,0],
        [0,-1,0],
        [0,0,-1],
        
        # [1,1,0],
        # [0,1,1],
        # [1,0,1],
        # [-1,-1,0],
        # [0,-1,-1],
        # [-1,0,-1],
        # [1,-1,0],
        # [0,1,-1],
        # [1,0,-1],
        # [-1,1,0],
        # [0,-1,1],
        # [-1,0,1],
        
        [1,1,1],
        
        [-1,1,1],
        [1,-1,1],
        [1,1,-1],
        
        [1,-1,-1],
        [-1,1,-1],
        [-1,-1,1],

        [-1,-1,-1],
        ])

    normal_position_nomalized = normal_position_nomalized / np.repeat(np.array([np.sum(np.abs(normal_position_nomalized)**2,axis=-1)**(1./2)]).T,3,axis=1)
    normal_position = scale_sphere * np.copy(normal_position_nomalized)
    normal_position_sphere = np.copy(normal_position)
    unit = scale_sphere
    normal = unit * np.copy(normal_position_nomalized)
    
    shift = nodes[:,only_half].mean()
    
    for i in range(nodes.shape[0]):
        node_sphere = nodes[i,:]
        if only_half is not None:
            if (node_sphere[only_half] > shift):
                continue # remove half of the infinitesimal elements
            node_sphere[only_half] = node_sphere[only_half] + shift # shift the remaining infinitesimal elements
            # node_sphere[only_half] = node_sphere[only_half] + nodes[:,only_half].mean() # shift the remaining infinitesimal elements
        x_sphere = x + node_sphere[0]
        y_sphere = y + node_sphere[1]
        z_sphere = z + node_sphere[2]
        normal_position_sphere = normal_position + np.tile(node_sphere, (normal_position.shape[0], 1))
        sig_sphere = sig_nodes[i,:,:]
        traction_sphere = np.matmul(sig_sphere,normal_position_nomalized.T).T
        ax.plot_surface(x_sphere,y_sphere,z_sphere,alpha=alpha,color=color)
        plot_vector_field(ax,normal_position_sphere,2 * unit * traction_sphere,linewidths=0.2,color=COLOR0)
    return ax

def plot_coo(ax,scale=.25,shift=-np.array([.25,.25,.25]),linewidths=0.5,fontsize=5,shifttxt=0.5):
    # plot coordinate system arrows
    if ax.name != "3d":
        e0 = np.array([0,0]) + shift[0:1]
        e1 = np.array([scale,0]); e1txt = e0 + (1+shifttxt)*e1
        e2 = np.array([0,scale]); e2txt = e0 + (1+shifttxt)*e2
        ax.quiver(e0[0],e0[1],e1[0],e1[1],scale=1,linewidths=linewidths,headwidth=2,headlength=1,headaxislength=1,color=COLOR4)
        ax.quiver(e0[0],e0[1],e2[0],e2[1],scale=1,linewidths=linewidths,headwidth=2,headlength=1,headaxislength=1,color=COLOR6)
        ax.text(e1txt[0],e1txt[1],r'$x_1$',size=fontsize,color=COLOR0,horizontalalignment='center',verticalalignment='center')
        ax.text(e2txt[0],e2txt[1],r'$x_2$',size=fontsize,color=COLOR0,horizontalalignment='center',verticalalignment='center')
    if ax.name == "3d":
        e0 = np.array([0,0,0]) + shift
        e1 = np.array([scale,0,0]); e1txt = e0 + (1+shifttxt)*e1
        e2 = np.array([0,scale,0]); e2txt = e0 + (1+shifttxt)*e2
        e3 = np.array([0,0,scale]); e3txt = e0 + (1+shifttxt)*e3
        ax.quiver(e0[0],e0[1],e0[2],e1[0],e1[1],e1[2],linewidths=1,colors=COLOR4)
        ax.quiver(e0[0],e0[1],e0[2],e2[0],e2[1],e2[2],linewidths=1,colors=COLOR5)
        ax.quiver(e0[0],e0[1],e0[2],e3[0],e3[1],e3[2],linewidths=1,colors=COLOR6)
        ax.text(e1txt[0],e1txt[1],e1txt[2],r'$x_1$',size=fontsize,color=COLOR0,horizontalalignment='center',verticalalignment='center')
        ax.text(e2txt[0],e2txt[1],e2txt[2],r'$x_2$',size=fontsize,color=COLOR0,horizontalalignment='center',verticalalignment='center')
        ax.text(e3txt[0],e3txt[1],e3txt[2],r'$x_3$',size=fontsize,color=COLOR0,horizontalalignment='center',verticalalignment='center')
    return ax






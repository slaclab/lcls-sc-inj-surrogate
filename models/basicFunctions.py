import numpy as np
import matplotlib.pyplot as plt
import sys, os
from IPython.display import clear_output
import h5py
import json
import tensorflow
from typing import List
    
#found online for axis formatting
import matplotlib.ticker as mticker

class MathTextSciFormatter(mticker.Formatter):
    def __init__(self, fmt="%1.2e"):
        self.fmt = fmt
    def __call__(self, x, pos=None):
        s = self.fmt % x
        decimal_point = '.'
        positive_sign = '+'
        tup = s.split('e')
        significand = tup[0].rstrip(decimal_point)
        sign = tup[1][0].replace(positive_sign, '')
        exponent = tup[1][1:].lstrip('0')
        if exponent:
            exponent = '10^{%s%s}' % (sign, exponent)
        if significand and exponent:
            s =  r'%s{\times}%s' % (significand, exponent)
        else:
            s =  r'%s%s' % (significand, exponent)
        return "${}$".format(s)
        
        
# define class for showing training plot - found online
class PlotLosses(tensorflow.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):
       
        
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        
        clear_output(wait=True)
        plt.plot(self.x, self.losses, label="loss")
        plt.plot(self.x, self.val_losses, label="val_loss")
        plt.legend()
        plt.ylabel('error')
        plt.xlabel('epoch')
        plt.show()

plot_losses = PlotLosses()


def image_scale(xdata):  
    x_scale = np.max(xdata)
    x_offset = 0
    scaled_x = ((xdata/x_scale)-x_offset)   
    return scaled_x, x_scale, x_offset
def image_unscale(image_values, image_scale, image_offset):  
    data_scaled = ((image_values+image_offset)*image_scale)  
    return data_scaled
def get_scale(data):
    # helper used to get the mean and range of the data set
    offset = np.nanmin(data,axis=0)
    scale= np.nanmax(data,axis=0) - np.nanmin(data, axis=0)
    if scale == 0:
        scale = 1
    return offset, scale

def scale_data(data,offset, scale, lower, upper):
    # for mean 0 and std 1 data=(data-data.mean(axis=0))/data.std(axis=0)
    data_scaled=lower+((data-offset)*(upper-lower)/scale)
    return data_scaled

def unscale_data(data,offset,scale,lower,upper):
    data_unscaled=(((data-lower)*scale)/(upper-lower)) + offset
    return data_unscaled

def compare_data_out(y1_,y2_,set_names_list, var_list=['rms_x', 'rms_y', 'rms_s', 'emit_x', 'emit_y', 'emit_s', 'dE', 'energy','numParticles','s'], fname='compare'):
    plt.close()
    fig = plt.figure()
    for i in range(0,y1_.shape[1]):
        indx=np.argsort(y1_[:,i])
        temp_plt = y1_[indx,i]
        temp_plt2 = y2_[indx,i]
        #plt.subplot(6,2,i+1)
        plt.plot(temp_plt, 'k.')
        plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
        plt.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.e"))
        plt.plot(temp_plt2, 'r+', markersize = .75)
        plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%1.1e"))
        plt.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.e"))
        plt.legend(set_names_list)
        plt.title(var_list[i])
        plt.xlabel('Sample Number')
        fig.set_figheight(4)
        fig.set_figwidth(2)
        plt.show()

    plt.show()
    plt.close()



def MakeAllHistograms(inputs,predictions, xnames, ynames, nbins = 50):
    '''Makes Histograms of all inputs and outputs, with column names given as xnames and ynames'''
    print('------------------------ Histograms of Input Parameter Values ------------------------')
    for i in range(len(xnames)):
        plt.figure(num=None, figsize=(10, 3), dpi=80, facecolor='w', edgecolor='k')
        plt.hist(inputs[:,i],bins = nbins)
        plt.title(str(xnames[i])+' Histogram')
        plt.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.2e"))
        plt.xlabel(str(xnames[i]))
        plt.show()
    

    print('------------------------ Histograms of Prediction Values ------------------------')
    for i in range(len(ynames)):
        plt.figure(num=None, figsize=(10, 3), dpi=80, facecolor='w', edgecolor='k')
        plt.hist(predictions[:,i],bins = nbins)
        plt.title(str(ynames[i])+' Histogram')
        plt.gca().xaxis.set_major_formatter(MathTextSciFormatter("%1.2e"))
        plt.xlabel(str(ynames[i]))
        plt.show()
        
def MakeProjection(predictions, xind, yind, xname, yname, xscaler = 1, yscaler = 1, plotsize = 10):
    '''Makes projection plots (phase space) for two predictions. Supply indices for chosen parameters
        and the name of each parameter (x first, then y). Optional scaling for axes can be provided.
        Optional plot size can be given'''
    plt.figure(num=None, figsize=(plotsize, plotsize), dpi=80, facecolor='w', edgecolor='k')
    plt.plot(predictions[:,xind]*xscaler,predictions[:,yind]*yscaler, 'k.')
    plt.title('Projection Plot for '+str(xname)+' and '+str(yname))
    plt.xlabel(xname)
    plt.ylabel(yname)
    plt.gca().yaxis.set_major_formatter(MathTextSciFormatter("%2.2e"))
    plt.gca().xaxis.set_major_formatter(MathTextSciFormatter("%2.2e"))
    plt.show()
    
def cull1(xarray, yarray, xref, yref, index_array):
    """
    Simple culling of dominated values. Assumes weights -1, -1 (minimization)
    """
    select = np.logical_or( xarray <= xref ,  yarray <= yref)
    return np.extract(select, xarray), np.extract(select, yarray), np.extract(select, index_array)

def cull(xarray, yarray, xreflist, yreflist):
    xout = np.array(xarray)
    yout =  np.array(yarray)
    iout =  np.arange(len(yarray))
    for x0, y0, in zip(xreflist, yreflist):
         xout, yout, iout = cull1(xout, yout, x0, y0, iout)
    return xout, yout, iout

def fast_pareto_2d_with_indices(xarray, yarray):
    """
    Returns x_front, y_front, indices
    """
    return cull(xarray, yarray, xarray, yarray)    

def image_scale(xdata):  
    x_scale = np.max(xdata)
    x_offset = 0
    scaled_x = ((xdata/x_scale)-x_offset)   
    return scaled_x, x_scale, x_offset
def image_unscale(image_values, image_scale, image_offset):  
    data_scaled = ((image_values+image_offset)*image_scale)  
    return data_scaled

def do_scaling(xdata,lower,upper):  
    l,n = xdata.shape
    
    x_scales = []
    x_offsets = []

    scaled_x = np.zeros((l,n))

    for i in range(n):
        dat = xdata[:,i]
        off, sc = get_scale(dat)
        x_offsets.append(off)
        x_scales.append(sc)
        scaled = scale_data(dat,off,sc,lower,upper)
        scaled_x[:,i] = scaled
        
    return scaled_x, x_scales, x_offsets

def do_unscaling(data, lower, upper, scales, offsets):
    n,m = np.shape(data)
    unscaled = np.zeros((n,m))
    for i in range(m):
        dat= data[:,i]
        sc = scales[i]
        off = offsets[i]
        sc_back = unscale_data(dat,off,sc,lower,upper)
        unscaled[:,i] = sc_back
    return unscaled
    
def fstr(s):
    """
    Makes a fixed string for h5 files
    """
    return np.string_(s)
def mySaveModel(filename,
               description,
               archjson,
               inputsc,
               inputoff,
               outputsc,
               outputoff,
               lower = -1,
               upper = 1,
               JSONFile = True):
    # Everything will be saved into this h5
    h = h5py.File(filename,'a')
    
    # Contains model architecture as JSON string 
    if JSONFile:
        f = open(archjson,'r')
        json_string = f.read()
        h.attrs.create("JSON", fstr(json_string))
    else:
        h.attrs.create("JSON", fstr(archjson))
        
    with open(description) as d:
        desc_file = dict(json.load(d))
        
    h.attrs.create("input_scales", inputsc)
    h.attrs.create("input_offsets", inputoff)
    h.attrs.create("lower", lower)
    h.attrs.create("upper", upper)
    h.attrs.create("input_names", list(desc_file["input_units"].keys()))
    h.attrs.create("input_ranges", list(desc_file["input_ranges"].values()))
    h.attrs.create("input_units", list(desc_file["input_units"].values()))
    h.attrs.create("input_ordering", desc_file["input_ordering"])

    if desc_file["type"] == "image":
        print("Type is image")
        h.attrs.create("type", "image")
        h.attrs.create("bins", desc_file["bins"])
        h.attrs.create("ndim", desc_file["ndim"])
        h.attrs.create("output_scales", outputsc)
        h.attrs.create("output_offsets", outputoff)
    elif desc_file["type"] == "both":
        print("Type is both")
        h.attrs.create("type", "both")
        h.attrs.create("bins", desc_file["bins"])
        h.attrs.create("ndim", desc_file["ndim"])
        h.attrs.create("output_scales", outputsc)
        h.attrs.create("output_offsets", outputoff)
        h.attrs.create("output_names", list(desc_file['output_units'].keys()))
        h.attrs.create("output_units", list(desc_file['output_units'].values()))
        h.attrs.create("output_ordering", desc_file['output_ordering'])
    else:
        print("Type is scalar")
        h.attrs.create("type", "scalar")
        h.attrs.create("output_scales", outputsc)
        h.attrs.create("output_offsets", outputoff)
        h.attrs.create("output_names", list(desc_file['output_units'].keys()))
        h.attrs.create("output_units", list(desc_file['output_units'].values()))
        h.attrs.create("output_ordering", desc_file['output_ordering'])
    h.close()

def makeDict(names, units, default_val = None, ranges = None):
    di = {}
    if ranges and default_val:
        for ind, name in enumerate(names): 
            d = {}
            d['variable_type'] = 'scalar'
            d['units'] = units[ind]
            d['default'] = default_val[ind]
            d['range'] = ranges[ind]
            di[name] = d
    else:
        for ind, name in enumerate(names): 
            d = {}
            d['variable_type'] = 'scalar'
            d['units'] = units[ind]
            di[name] = d
    return di

def makeDescription(
    name, 
    description, 
    input_names,
    input_units,
    input_defaults, 
    input_ranges,
    output_names, 
    output_units):
    
    d = {}
    d['name'] = name
    d['description'] = description
    ## make inputs dict
    di = makeDict(input_names, input_units, input_defaults, input_ranges)
    d['input_variables'] = di
    d['input_variables']['input_image'] =  {'variable_type': 'image','axis_labels': ['x', 'y'],'range': [0, 10],'x_min_variable': 'in_xmin','x_max_variable': 'in_xmax','y_min_variable': 'in_ymin','y_max_variable': 'in_ymax'}
    image_in_extents = ['in_xmin', 'in_xmax', 'in_ymin', 'in_ymax']
    for ex in image_in_extents:
        d['input_variables'][ex]['parent_variable'] = 'input_image'
        
    do = makeDict(output_names, output_units)
    d['output_variables'] = do
    d['output_variables']['x:y'] = {'variable_type': 'image','axis_units': ['mm', 'mm'],'axis_labels': ['x', 'y'],'x_min_variable': 'out_xmin','x_max_variable': 'out_xmax','y_min_variable': 'out_ymin','y_max_variable': 'out_ymax'}
    image_out_extents = ['out_xmin', 'out_xmax', 'out_ymin', 'out_ymax']
    for ex in image_out_extents:
        d['output_variables'][ex]['parent_variable'] = 'x:y'
        
    return d
    
def save_model(
    weight_file: str,  # model
    description_file: str,
    arch_json: str,
    input_scales: List[float],
    input_offsets: List[float],
    output_scales: List[float],
    output_offsets: List[float],
    lower: float = 0,
    upper: float = 1,
    arch_json_file: bool = True,
):
    """
    Utility function for saving the model.
    Args:
        weight_file (str): Model weight filename.
        description_file (str): Description filename.
        input_scales (List[float]): List of values used for input scaling.
        input_offsets (List[float]): Offsets for input variables.
        output_scales (List[float]): List of values used for output scaling.
        output_offsets (List[float]): Offsets for output variables.
        lower (float): Lower bound on variable scaling.
        upper (float): Upper bound on variable scaling.
        arch_json (str): JSON representation of model architecture.
        arch_json_file (str): File holding JSON representation of model architecture.
        input_variable_extras (dict): Additional keys used for building input variables.
        output_variable_extras (dict): Additional keys used for building output variables.
        variable_filename (str): File for saving variables.
    """

    # Everything will be saved into this h5
    h = h5py.File(weight_file, "a")

    # Contains model architecture as JSON string
    if arch_json_file:
        f = open(arch_json, "r")
        json_string = f.read()
        h.attrs.create("JSON", fstr(json_string))
    else:
        h.attrs.create("JSON", fstr(arch_json))

    with open(description_file, 'r') as d:
        description = dict(json.load(d))

    h.attrs.create("input_ordering", list(description["input_variables"].keys()))
    h.attrs.create("output_ordering", list(description["output_variables"].keys()))
    h.attrs.create("input_scales", input_scales)
    h.attrs.create("input_offsets", input_offsets)
    h.attrs.create("output_scales", output_scales)
    h.attrs.create("output_offsets", output_offsets)
    h.attrs.create("lower", lower)
    h.attrs.create("upper", upper)
    h.close()


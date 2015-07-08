# imports and basic notebook setup
from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import cv2
from IPython.display import clear_output, Image, display
from google.protobuf import text_format
import os
from IPython.core.debugger import Tracer

import caffe
caffe.set_mode_cpu()

# ## Loading DNN model
net_fn   = os.path.expanduser('~') + '/research/deep_context/proto/dextro_277_network_structure.prototxt'
param_fn = os.path.expanduser('~') + '/research/deep_context/models/dextro_277_trained_network.model'
mean_fn  = os.path.expanduser('~') + '/research/deep_context/data/mean_image.npy'

# Patching model to be able to compute gradients.
# Note that you can also manually add "force_backward: true" line to "deploy.prototxt".
model = caffe.io.caffe_pb2.NetParameter()
text_format.Merge(open(net_fn).read(), model)
model.force_backward = True
open('tmp.prototxt', 'w').write(str(model))

net = caffe.Net('tmp.prototxt', param_fn, caffe.TEST)
mean_image = np.load(mean_fn)
mean_pixel = mean_image.mean(axis=0).mean(axis=0)

INNER_VIS = False
OUTER_VIS = False

# a couple of utility functions for converting to and from Caffe's input image layout
def preprocess(im):
    return (im - mean_pixel).transpose((2, 0, 1)).astype(float)

def deprocess(caffe_out):
    return (caffe_out[0].transpose((1, 2, 0)) + mean_pixel).astype('uint8')

###  Producing dreams
def make_step(net, step_size=1.5, end='pool5', jitter=32, clip=True):
    '''Basic gradient ascent step.'''

    src = net.blobs['data'] # input image is stored in Net's 'data' blob
    dst = net.blobs[end]

    ox, oy = np.random.randint(-jitter, jitter+1, 2)
    src.data[0] = np.roll(np.roll(src.data[0], ox, axis=-1), oy, axis=-2) # apply jitter shift
            
    #net.forward(end=end)
    net.forward(end=end)
    dst.diff[:] = dst.data  # specify the optimization objective
    net.backward(start=end)
    g = src.diff[0]
    # apply normalized ascent step to the input image
    src.data[:] += step_size/np.abs(g).mean() * g

    src.data[0] = np.roll(np.roll(src.data[0], -ox, axis=-1), -oy, axis=-2) # unshift image
            
    if clip:
        src.data[0, 0, :, :] = np.clip(src.data[0, 0, :, :], -mean_pixel[0], 255-mean_pixel[0])
        src.data[0, 1, :, :] = np.clip(src.data[0, 1, :, :], -mean_pixel[1], 255-mean_pixel[1])
        src.data[0, 2, :, :] = np.clip(src.data[0, 2, :, :], -mean_pixel[2], 255-mean_pixel[2])

# Next we implement an ascent through different scales. We call these scales "octaves".
def deepdream(net, base_img, iter_n=7, octave_n=4, octave_scale=1.4, end='pool5', clip=True, **step_params):
    # prepare base images for all octaves
    octaves = [preprocess(base_img)]
    for i in xrange(octave_n-1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0/octave_scale,1.0/octave_scale), order=1))
    
    src = net.blobs['data']
    detail = np.zeros_like(octaves[-1]) # allocate image for network-produced details
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]
        if octave > 0:
            # upscale details from the previous octave
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0*h/h1,1.0*w/w1), order=1)

        src.reshape(1,3,h,w) # resize the network's input image size
        src.data[0] = octave_base+detail
        for i in xrange(iter_n):
            make_step(net, end=end, clip=clip, **step_params)
            
            # visualization
            #Tracer()()
            if INNER_VIS:
                vis = deprocess(src.data)
                if not clip: # adjust image contrast if clipping is disabled
                    vis = vis*(255.0/np.percentile(vis, 99.98))
                cv2.imshow('.', vis)
                cv2.waitKey(10)
            print octave, i, end, src.data[0].shape
            #clear_output(wait=True)
            
        # extract details produced on the current octave
        detail = src.data[0]-octave_base
    # returning the resulting image
    out = deprocess(src.data)
    if not clip:
        out = out*(255.9/np.percentile(out, 99.98))
    return out

def simple_dream(im, end_layer):
    im_d = deepdream(net, im.astype(float), end=end_layer)
    if OUTER_VIS:
        cv2.imshow('dream', im_d)
    return im_d

def complex_dream(im, end_layer):
    frame = im[:]
    h, w = frame.shape[:2]
    s = 0.05 # scale coefficient
    for i in xrange(3):
        frame = deepdream(net, frame.astype(float), end=end_layer)
        if OUTER_VIS:
            cv2.imshow('dream', frame)
            cv2.imwrite('dream_frames/%04d.jpg'%i, frame)
        frame = nd.affine_transform(frame, [1-s,1-s,1], [h*s/2,w*s/2,0], order=1)
    return frame

def video_dream(cap, wri, end_layer):
    if not cap.isOpened():
        print 'Could not open input video'
        return
    if not wri.isOpened():
        print 'Could not open output video'
        return

    count = 0
    while True:
        retval, im = cap.read()
        if not retval:
            break
        print 'Processing frame %d'%count
        im = cv2.resize(im, (int(im.shape[1]*1.5), int(im.shape[0]*1.5)))
        im_d = complex_dream(im, end_layer)
        count += 1
        cv2.imwrite('video_frames/%04d.jpg'%count, im_d)
        wri.write(im_d)

    cap.release()
    wri.release()
        
#im_filename = 'sky1024px.jpg'
#im = cv2.imread(im_filename)
#complex_dream(im)

end_layer = 'pool5'
vid_filename = 'dog_in.mp4'
cap = cv2.VideoCapture(vid_filename)
fourcc = cv2.cv.CV_FOURCC('X', 'V', 'I', 'D')
h = int(1.5 * cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
w = int(1.5 * cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
wri = cv2.VideoWriter('dog_out.avi', fourcc, 6, (w, h))
video_dream(cap, wri, end_layer)

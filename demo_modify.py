import os
import time
import cv2
import numpy as np
from skimage import segmentation
import torch
import torch.nn as nn
import argparse
import fiona
from rasterio.features import shapes
import rasterio as rio
import imageio
import geopandas as gpd
from rasterio import mask as msk

""" Sources:
segmentation:
- modified  script I extended: ( this works without GPU too, when there is no GPU available, runs on CPU):  https://github.com/Yonv1943/Unsupervised-Segmentation (demo_modify.py)
- original (does not wrk without GPU, needs to be rewritten for running on CPU):  https://github.com/kanezaki/pytorch-unsupervised-segmentation (demo.py)

polygonization:  https://github.com/rasterio/rasterio/blob/fb93a6425c17f25141ad308cee7263d0c491a0a9/examples/rasterio_polygonize.py#L9

"""

parser = argparse.ArgumentParser(description='PyTorch Unsupervised Segmentation')
parser.add_argument('--input','-i', metavar='FILENAME', type=str, help='input image file name (required)')
parser.add_argument('--MaxIter', type=int, default =64, help='Maximum number of iterations (optional)')
parser.add_argument('--no_visualization', dest='visualization', action = 'store_false', help='Do not show window, only save result image and polygons (optional)' ) # nem jeleníti meg a képet
parser.add_argument('--gif', dest='gif', action = 'store_true', help='Save intermediate steps to gif (optional)' ) # GIF export
parser.add_argument('--StopCriterium', '-sc',type=float, default = 0.05, help='(optional)')
parser.add_argument('--clip','-cl', metavar='FILENAME', type=str, help='polygons to clip with (optional)',default='')
parser.add_argument('--MaxPolygon', type=int, help='max_polygons to iterate over (optional)',default=0)
parser.add_argument('--FilterById',metavar='FIELD NAME', type=str, help='field name for selecting polygons (optional)',default='')
parser.add_argument('--FieldValue',metavar = 'FIELD VALUE', type=int, help='field value for selecting polygons (optional)',default=-1)
parser.set_defaults(visualization=True)
parser.set_defaults(gif=False)
arguments = parser.parse_args()



def polygonize(image, vector_file, transform, driver, crs, mask_value):
    
    #with rs.drivers(): ez nem kell, hibaüzenetet dob!
        
    if mask_value is not None:
        mask = image == mask_value
    else:
        mask = None
    
    results = [
        {'properties': {'raster_val': v}, 'geometry': s}
        for i, (s, v) 
        in enumerate(
            shapes(image, mask=mask, transform=transform))]

    with fiona.open(
            vector_file, 'w', 
            driver=driver,
            crs=crs,
            schema={'properties': [('raster_val', 'int')],
                    'geometry': 'Polygon'}) as dst:
        dst.writerecords(results)
    return dst.name


class Args(object):

    input_image_path = arguments.input
    train_epoch = arguments.MaxIter
    mod_dim1 = 64  #
    mod_dim2 = 32
    gpu_id = 0

    min_label_num = 4  # if the label number smaller than it, break loop
    max_label_num = 256  # if the label number smaller than it, start to show result image.


class MyNet(nn.Module):
    def __init__(self, inp_dim, mod_dim1, mod_dim2):
        super(MyNet, self).__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(inp_dim, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim2, mod_dim1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(mod_dim1),
            nn.ReLU(inplace=True),

            nn.Conv2d(mod_dim1, mod_dim2, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(mod_dim2),
        )

    def forward(self, x):
        return self.seq(x)


def run():

    args = Args()
    torch.cuda.manual_seed_all(1943)
    np.random.seed(1943)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)  # choose GPU:0
    
    def process(image,args,crs,geo_transform,idx):
        start_time0 = time.time()
        #args = Args()
        kr = arguments.StopCriterium
        if idx is not None:
            str_i = str(idx)
        else:
            str_i = ''
        if arguments.gif:
            if len(str_i)>1:
                gif = args.input_image_path.split('.')[0]+'_seg'+'s.'+'_'+str(arguments.MaxIter)+'_'+str_i+'.gif'  
            elif len(arguments.FilterById)>0 and arguments.FieldValue >0:
                gif = args.input_image_path.split('.')[0]+'_seg'+'s.'+'_'+str(arguments.MaxIter)+'_'+str(arguments.FieldValue)+'.gif'
            else:
                gif = args.input_image_path.split('.')[0]+'_seg'+'s.'+'_'+str(arguments.MaxIter)+'.gif'
            writer=imageio.get_writer(gif, mode="I",fps=1)

        '''segmentation ML'''
        seg_map = segmentation.felzenszwalb(image, scale=32, sigma=0.5, min_size=64)
        # seg_map = segmentation.slic(image, n_segments=10000, compactness=100)
        seg_map = seg_map.flatten()
        seg_lab = [np.where(seg_map == u_label)[0]
                for u_label in np.unique(seg_map)]
        

        '''train init'''
        device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

        tensor = image.transpose((2, 0, 1))
        tensor = tensor.astype(np.float32) / 255.0
        tensor = tensor[np.newaxis, :, :, :]
        tensor = torch.from_numpy(tensor).to(device)

        model = MyNet(inp_dim=3, mod_dim1=args.mod_dim1, mod_dim2=args.mod_dim2).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=5e-2, momentum=0.9)
        # optimizer = torch.optim.RMSprop(model.parameters(), lr=1e-1, momentum=0.0)

        image_flatten = image.reshape((-1, 3))
        color_avg = np.random.randint(255, size=(args.max_label_num, 3))
        show = image

        '''train loop'''
        start_time1 = time.time()
        model.train()
        inhomo = list()
        for batch_idx in range(args.train_epoch):
            '''forward'''
            optimizer.zero_grad()
            output = model(tensor)[0]
            output = output.permute(1, 2, 0).view(-1, args.mod_dim2)
            target = torch.argmax(output, 1)
            im_target = target.data.cpu().numpy()

            '''refine'''
            for inds in seg_lab:
                u_labels, hist = np.unique(im_target[inds], return_counts=True)
                im_target[inds] = u_labels[np.argmax(hist)]

            '''backward'''
            target = torch.from_numpy(im_target)
            target = target.to(device)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            '''show image'''
            
            un_label, lab_inverse = np.unique(im_target, return_inverse=True, )
            if un_label.shape[0] < args.max_label_num:  # update show
                img_flatten = image_flatten.copy()
                if len(color_avg) != un_label.shape[0]:
                    color_avg = [np.mean(img_flatten[im_target == label], axis=0, dtype=np.int32) for label in un_label]
                for lab_id, color in enumerate(color_avg):
                    img_flatten[lab_inverse == lab_id] = color
                show = img_flatten.reshape(image.shape)
                gray_image = cv2.cvtColor(show, cv2.COLOR_BGR2GRAY)
                print('unique values:',end=' ')
                u=np. unique(gray_image, return_counts=True)
                print(len(u[0]),end=' ')
                print('Inhomogeneity:',end=' ')
                inh=1e4*np.sum(np.sqrt(u[1]))/(gray_image.shape[0]*gray_image.shape[0])
                print(inh)
                inhomo.append(inh)
                if batch_idx>3:
                    if abs(inhomo[-1]-inhomo[-2]) < kr and abs(inhomo[-2]-inhomo[-3]) < kr:
                        break
                if batch_idx ==0:
                    kr = inh /100 *kr

            if arguments.visualization:
                cv2.imshow("seg_pt", show)
                cv2.waitKey(1)
            if arguments.gif :
                rgb_frame = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
                writer.append_data(rgb_frame)

            print('Loss:', batch_idx, loss.item())
            if len(un_label) < args.min_label_num:
                break

        '''save'''
        time0 = time.time() - start_time0
        time1 = time.time() - start_time1
        print('PyTorchInit: %.2f\nTimeUsed: %.2f' % (time0, time1))

        if len(str_i)>1:
            out = args.input_image_path.split('.')[0]+'_seg'+str(int(time1))+'s_'+str(arguments.MaxIter)+'_'+str_i+'.'+args.input_image_path.split('.')[1]
            out_vector = args.input_image_path.split('.')[0]+'_seg'+str(int(time1))+'s_'+str(arguments.MaxIter)+'_'+str_i+'.shp'
        elif len(arguments.FilterById)>0 and arguments.FieldValue >0:
            out = args.input_image_path.split('.')[0]+'_seg'+str(int(time1))+'s_'+str(arguments.MaxIter)+'_'+str(arguments.FieldValue)+'.'+args.input_image_path.split('.')[1]
            out_vector = args.input_image_path.split('.')[0]+'_seg'+str(int(time1))+'s_'+str(arguments.MaxIter)+'_'+str(arguments.FieldValue)+'.shp'
        else:
            out = args.input_image_path.split('.')[0]+'_seg'+str(int(time1))+'s_'+str(arguments.MaxIter)+'.'+args.input_image_path.split('.')[1]
            out_vector = args.input_image_path.split('.')[0]+'_seg'+str(int(time1))+'s_'+str(arguments.MaxIter)+'.shp'
        print(out)
        cv2.imwrite(out, show)
        if arguments.gif :
            writer.close()

        name=polygonize(gray_image,out_vector,geo_transform,'ESRI Shapefile', crs,None)
        print(f'Vector geometry written to file: {name}.shp')


    if len(arguments.clip)>0:
        shapes = gpd.read_file(arguments.clip)
        if len(arguments.FilterById)>0 and arguments.FieldValue >0:
            with rio.open(args.input_image_path) as src:
                crs = src.crs
                shape = shapes.geometry[shapes[arguments.FilterById]==arguments.FieldValue].reset_index(drop=True)[0]
                image, geo_transform = msk.mask(src, [shape], crop=True)
                image = np.moveaxis(image,0,-1)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                process(image,args,crs,geo_transform,None)
        else:
            with rio.open(args.input_image_path) as src:
                crs = src.crs
                processed_polygons=0
                for i,shape in enumerate(shapes.geometry):
                    try:
                        image, geo_transform = msk.mask(src, [shape], crop=True)
                        image = np.moveaxis(image,0,-1)
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                        process(image,args,crs,geo_transform,i)
                        processed_polygons+=1
                    except:
                        continue
                    if processed_polygons >= arguments.MaxPolygon:
                        break
    else:
        image = cv2.imread(args.input_image_path)
        with rio.open(arguments.input) as src:
            geo_transform = src.transform
            crs = src.crs
        process(image,args,crs,geo_transform,None)


if __name__ == '__main__':
    run()

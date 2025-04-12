## USAGE: python3 figo-tasm_video-object_query.py --video ../data/birds.mp4 --label bird --labelcount 1 --out query_out/
############### FiGO-TASM integration
import os
import argparse, shutil, json, cv2, time
import tasm
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from glob import glob
from loader.image_loader import ImageLoader
from predicate.count_predicate import CountPredicate
from evaluator.f1_evaluator import F1Evaluator
from naive.scheduler import Scheduler as NaiveScheduler
from msfilter.scheduler import Scheduler as MSFilterScheduler
from mecoarse.scheduler import Scheduler as MECoarseScheduler
from figo.scheduler import Scheduler as FiGOScheduler
from mc.scheduler import Scheduler as MCScheduler
from profiler.model_time_titanx import setup_static_model_time

parser = argparse.ArgumentParser(description="A simple argparse example")
parser.add_argument('--video', type=str, required=True, help='Video path')
parser.add_argument('--label', type=str, required=True, help='object label')
parser.add_argument('--labelcount', type=int, default=1, help='Your age')
parser.add_argument('--out', type=str, required=True, help='Query output directory')
args = parser.parse_args()
# args = parser.parse_args(['--video', '../data/birds.mp4', '--label', 'car', '--labelcount', '1', '--out', 'query_out/'])
print(f"##### INPUTS: \nVideo:{args.video}\nObject-label:{args.label}\nmin_Object-count:{args.labelcount}\nQuery_output:{args.out}")

############### Setup environment
NOTEBOOK_RESOURCES_PATH = 'basics_resources'
TASM_STATUS_PATH = NOTEBOOK_RESOURCES_PATH+'/processed_videos_inTASM.json'
# os.system('rm '+TASM_STATUS_PATH)
if os.path.exists(NOTEBOOK_RESOURCES_PATH):
    if(os.path.exists(TASM_STATUS_PATH)):
        processed_videos = json.load(open(TASM_STATUS_PATH, 'r'))
        # print(processed_videos)
    else:
        processed_videos = {}
        ### to remake db and catalog paths
        shutil.rmtree(NOTEBOOK_RESOURCES_PATH)
        os.mkdir(NOTEBOOK_RESOURCES_PATH)
else:
    os.mkdir(NOTEBOOK_RESOURCES_PATH)
    processed_videos = {}

tasm.configure_environment({
    'default_db_path': os.path.join(NOTEBOOK_RESOURCES_PATH, 'labels.db'),
    'catalog_path': os.path.join(NOTEBOOK_RESOURCES_PATH, 'resources') 
})
t = tasm.TASM()

############### Inputs:
VIDEO_PATH = args.video #'../data/birds.mp4'
label = args.label #'bird'
tile_format = '-untiled'
# tile_format = '-2x2'      #TODO: currently object overlapping on multiple tiles is not supported in modified TASM
# tile_format = '-2x4'      #TODO: currently object overlapping on multiple tiles is not supported in modified TASM
### tile_format = '-birds'  #TODO: currently not supported in FiGO-TASM integration
first_frame_inclusive = 0
last_frame_exclusive = 180    #TODO: --> Resolve error of object appearing in multiple tiles
if('x' in tile_format):
    rows, cols = tile_format[1:].split('x')
    rows, cols = int(rows), int(cols)
metadata_id = VIDEO_PATH.split('/')[-1][:-4] #'birds'
tiled_video_name = metadata_id + tile_format

### Store Video in TASM
store_video_inTASM = False
if(metadata_id not in processed_videos):
    processed_videos[metadata_id] = {'TASM_video_store':[], 'TASM_indexed_objects':[], 'FiGO_scanned_objects':[], 'FiGO_removed_objects':[]}
    store_video_inTASM = True
else:
    if(tile_format not in processed_videos[metadata_id]['TASM_video_store']):
        store_video_inTASM = True
    else:
        print("#### Video already stored in TASM with tile layout:",tile_format)
if(store_video_inTASM):
    print("#### Storing video in TASM with tile layout:",tile_format)
    if('untiled' in tile_format):
        t.store(VIDEO_PATH, tiled_video_name)
        processed_videos[metadata_id]['TASM_video_store'].append(tile_format)
    elif(f'{rows}x{cols}' in tile_format):
        t.store_with_uniform_layout(VIDEO_PATH, metadata_id+'-'+f'{rows}x{cols}', rows, cols)
        processed_videos[metadata_id]['TASM_video_store'].append(tile_format)
    else:
        ### TODO: if non-uniform tile layout store video after obtaining metadata
        pass
    json.dump(processed_videos, open(TASM_STATUS_PATH, 'w'), indent=4)
    print("#### Storage of video in TASM complete with tile layout:",tile_format)

### Output storage for Input Query
query_out_dir = args.out+'/'+metadata_id+'/'
# query_out_dir = 'query_out/'+metadata_id+'/'
os.makedirs(query_out_dir,exist_ok=True)
os.system('rm '+query_out_dir+'/frame1*.jpg')
os.system('rm '+query_out_dir+'/frame2*.jpg')
os.system('rm '+query_out_dir+'/frame3*.jpg')
os.system('rm '+query_out_dir+'/frame4*.jpg')
os.system('rm '+query_out_dir+'/frame5*.jpg')
os.system('rm '+query_out_dir+'/*.jpg')
query_out_df = {'frame':[],'label':[], 'x1':[], 'y1':[], 'width':[], 'height':[]}
query_out_csv_path = query_out_dir+'query_metadata.csv'
query_profile_df = {'video':[metadata_id],'label':[label], 'n_objects':[], 'n_object_frames':[], 'FiGO_time':[], 'FiGO_TASMst_time':[], 'total_frames':[]}
query_profile_csv_path = query_out_dir+'query_profiling.csv'

print("Query data:", processed_videos[metadata_id])
### Process Query
if(label in processed_videos[metadata_id]['TASM_indexed_objects']):
    print("#### Video-Metadata already stored in TASM semantic index.")
    print("#### Process Video-query through TASM")
    
    startt = time.time()
    ####TODO: --> Resolve error of object appearing in multiple tiles
    # tiled_selection = t.select(tiled_video_name, metadata_id, label, first_frame_inclusive, last_frame_exclusive)
    tiled_selection = t.select(tiled_video_name, metadata_id, label)
    while True:
        bird = tiled_selection.next()
        if bird.is_empty():
            break
        query_out_df['frame'].append(bird.frame_id())
        query_out_df['label'].append(label)
        query_out_df['x1'].append(bird.frame_x())
        query_out_df['y1'].append(bird.frame_y())
        query_out_df['width'].append(bird.width())
        query_out_df['height'].append(bird.height())
        bird_img_filename = query_out_dir+'frame'+str(query_out_df['frame'][-1])+'_'+str(query_out_df['label'][-1])+'_x'+str(query_out_df['x1'][-1])+'_y'+str(query_out_df['y1'][-1])+'_w'+str(query_out_df['width'][-1])+'_h'+str(query_out_df['height'][-1])+'.jpg'
        cv2.imwrite(bird_img_filename, bird.numpy_array()[:,:,::-1])
    query_out_df = pd.DataFrame(query_out_df)
    query_out_df.to_csv(query_out_csv_path, index=None)
    duration_ms = time.time() - startt
    print(f"######## Total time for extracting video frames and metadata from {tiled_video_name} TASM tiling layout: {duration_ms:.4f} s")
    print(f'Detected {query_out_df.shape[0]} {label}. Saved query output images to "{query_out_dir}" with metadata in "{query_out_csv_path}".\n\n')
elif(label in processed_videos[metadata_id]['FiGO_scanned_objects'] or len(processed_videos[metadata_id]['FiGO_scanned_objects'])==0):
    print("#### Video-Metadata does not exist in TASM; Processing Video-query through FiGO.")
    total_FiGO_exec_time = 0
    startt = time.time()
    ### Prepare data for FiGO query execution
    dataset_type = 'newds'
    output_dir = 'data/'+dataset_type
    os.makedirs(output_dir, exist_ok=True)
    os.system('rm '+output_dir+'/*')
    cap = cv2.VideoCapture(VIDEO_PATH)
    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_filename = os.path.join(output_dir, f'frame_{frame_count:05d}.jpg')
        cv2.imwrite(frame_filename, frame)
        frame_count += 1
    cap.release()
    print(f"Saved {frame_count} frames to '{output_dir}'")
    duration_ms = time.time() - startt
    query_profile_df['total_frames'].append(frame_count)
    print(f"Total time to Prepare data for FiGO query execution: {duration_ms:.4f} s")
    ### Assuming dataset would be already prepared data prep time should not be added in FiGO query time
    # total_FiGO_exec_time+=duration_ms
    
    ### Prepare preicate/query and execute FiGO
    pred = CountPredicate([label], [args.labelcount])
    loader = ImageLoader(dataset_type, False)  #args.use_cache
    setup_static_model_time()  # if args.use_titanx_model_time:
    
    sched_f = FiGOScheduler
    sched = sched_f('efficientdet', loader, pred, False)  #args.use_cache
    ### TODO: disable storing image frames for long videos that would not fit in cpu memory, Modify: FiGO/modeling/efficientdet/model_wrapper.py
    res = sched.process()
    print("FiGO query time:",sched.get_query_time())
    total_FiGO_exec_time+=sched.get_query_time()
    
    ### Save query output
    all_video_frameimgs = sorted(glob(output_dir+'/*.jpg'))
    sorted_frame_idxs = sorted(res.keys(), key=int)
    metadata_info = []
    startt = time.time()
    all_objects = []
    for rki in sorted_frame_idxs:
        # if('ori_imgs' in res[rki]):
        #     frame_img = res[rki]['ori_imgs'][0]
        # else:
        #     frame_img = cv2.imread(all_video_frameimgs[rki])
        frame_img = cv2.imread(all_video_frameimgs[rki])
        for obji in range(len(res[rki]['class'])):
            if(res[rki]['class'][obji] not in all_objects):
                all_objects.append(res[rki]['class'][obji])
            if(label==res[rki]['class'][obji]):
                metadata_info.append(tasm.MetadataInfo(metadata_id, res[rki]['class'][obji], rki, res[rki]['roi'][obji][0], res[rki]['roi'][obji][1], res[rki]['roi'][obji][2], res[rki]['roi'][obji][3]))
                query_out_df['frame'].append(rki)
                query_out_df['label'].append(label)
                query_out_df['x1'].append(res[rki]['roi'][obji][0])
                query_out_df['y1'].append(res[rki]['roi'][obji][1])
                query_out_df['width'].append(res[rki]['roi'][obji][2]-res[rki]['roi'][obji][0])
                query_out_df['height'].append(res[rki]['roi'][obji][3]-res[rki]['roi'][obji][1])
                bird_img_filename = query_out_dir+'frame'+str(query_out_df['frame'][-1])+'_'+str(query_out_df['label'][-1])+'_x'+str(query_out_df['x1'][-1])+'_y'+str(query_out_df['y1'][-1])+'_w'+str(query_out_df['width'][-1])+'_h'+str(query_out_df['height'][-1])+'.jpg'
                cropped_bird_image = frame_img[query_out_df['y1'][-1]: query_out_df['y1'][-1]+query_out_df['height'][-1], query_out_df['x1'][-1]: query_out_df['x1'][-1]+query_out_df['width'][-1], ::-1]
                cv2.imwrite(bird_img_filename, cropped_bird_image)
                # print(cropped_bird_image.shape)
            else:
                ### This FiGO query is optimized for the specified object label in the query/predicate for other objects there might be optimal models to rely on when using FiGO
                pass
    duration_ms = time.time() - startt
    print(f"Total time for extracting video frames and metadata from FiGO query output: {duration_ms:.4f} s")
    total_FiGO_exec_time+=duration_ms
    query_profile_df['n_objects'].append(len(metadata_info))
    if(len(metadata_info)>0):
        query_profile_df['n_object_frames'].append(np.unique(query_out_df['frame']).shape[0])
        startt = time.time()
        t.add_bulk_metadata(metadata_info)
        processed_videos[metadata_id]['TASM_indexed_objects'].append(label)
        
        query_out_df = pd.DataFrame(query_out_df)
        query_out_df.to_csv(query_out_csv_path, index=None)
        duration_ms = time.time() - startt
        total_FiGO_exec_time+=duration_ms
        print(f'Detected {query_out_df.shape[0]} {label}. Saved query output images to "{query_out_dir}" with metadata in "{query_out_csv_path}".')
    else:
        query_profile_df['n_object_frames'].append(0)
        print(f'Detected 0 {label} in video. Removing it from scanned objects list')
        processed_videos[metadata_id]['FiGO_scanned_objects'].remove(label)
        processed_videos[metadata_id]['FiGO_removed_objects'].append(label)
        
    ### Store all figo scanned unique objects list into json
    if(len(processed_videos[metadata_id]['FiGO_scanned_objects'])==0):
        processed_videos[metadata_id]['FiGO_scanned_objects'] = all_objects
    else: 
        ### if any new object detected which is not in removed objs list then update the list by taking the union
        for alo in np.unique(all_objects):
            if(alo not in processed_videos[metadata_id]['FiGO_scanned_objects'] and alo not in processed_videos[metadata_id]['FiGO_removed_objects']):
                processed_videos[metadata_id]['FiGO_scanned_objects'].append(alo)
    json.dump(processed_videos, open(TASM_STATUS_PATH, 'w'), indent=4)
    print("Video-Metadata now stored in TASM semantic index.")
    print(f"######## Total time for executing query through FiGO: {total_FiGO_exec_time:.4f} s")
    query_profile_df['FiGO_time'].append(total_FiGO_exec_time)
    print("After Query updated data:", processed_videos[metadata_id], "\n\n")
    
    ### Rerunning for profiling TASM time
    query_out_df = {'frame':[],'label':[], 'x1':[], 'y1':[], 'width':[], 'height':[]}
    startt = time.time()
    tiled_selection = t.select(tiled_video_name, metadata_id, label)
    while True:
        bird = tiled_selection.next()
        if bird.is_empty():
            break
        query_out_df['frame'].append(bird.frame_id())
        query_out_df['label'].append(label)
        query_out_df['x1'].append(bird.frame_x())
        query_out_df['y1'].append(bird.frame_y())
        query_out_df['width'].append(bird.width())
        query_out_df['height'].append(bird.height())
        bird_img_filename = query_out_dir+'frame'+str(query_out_df['frame'][-1])+'_'+str(query_out_df['label'][-1])+'_x'+str(query_out_df['x1'][-1])+'_y'+str(query_out_df['y1'][-1])+'_w'+str(query_out_df['width'][-1])+'_h'+str(query_out_df['height'][-1])+'.jpg'
        cv2.imwrite(bird_img_filename, bird.numpy_array()[:,:,::-1])
    query_out_df = pd.DataFrame(query_out_df)
    query_out_df.to_csv(query_out_csv_path, index=None)
    duration_ms = time.time() - startt
    print(f"######## Total time for extracting video frames and metadata from {tiled_video_name} TASM tiling layout: {duration_ms:.4f} s")
    query_profile_df['FiGO_TASMst_time'].append(duration_ms)
    query_profile_df = pd.DataFrame(query_profile_df)
    query_profile_df.to_csv(query_profile_csv_path, index=None)
else:
    print(f"\n**** Queried object: {label} is not present in video! ****\n\n")
    
# Total time for extracting video frames and metadata from birds-untiled TASM tiling layout: 0.42 - 0.47 s

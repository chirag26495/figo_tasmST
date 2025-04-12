import os, json, pandas as pd

NOTEBOOK_RESOURCES_PATH = 'basics_resources'
TASM_STATUS_PATH = NOTEBOOK_RESOURCES_PATH+'/processed_videos_inTASM.json'
QUERY_OUT_DIR = 'query_out'
RUN_LOGS_PATH = 'execution_logs.log'
# os.system('rm -r '+NOTEBOOK_RESOURCES_PATH+'/*')
# os.system('rm -r '+QUERY_OUT_DIR+'/*')
# os.system('>'+RUN_LOGS_PATH)

def allobjectqueries(VIDEO_PATH, start_label):
    print(f"\n##### Query running for object: {start_label} in video {VIDEO_PATH}")
    os.system(f"python3 figo-tasm_video-object_query.py --video '{VIDEO_PATH}' --label '{start_label}' --out {QUERY_OUT_DIR}/ >> {RUN_LOGS_PATH}")
    
    metadata_id = VIDEO_PATH.split('/')[-1][:-4]
    query_profile_csv_path = QUERY_OUT_DIR+'/'+metadata_id+'/query_profiling.csv'
    query_profile_all_csv_path = query_profile_csv_path.replace('.csv','_all.csv')
    if(not os.path.exists(query_profile_all_csv_path)):
        os.system('cp '+query_profile_csv_path+' '+query_profile_all_csv_path)
    query_profile_all_df = pd.read_csv(query_profile_all_csv_path)
    while(True):
        processed_videos = json.load(open(TASM_STATUS_PATH, 'r'))
        if(len(processed_videos[metadata_id]['TASM_indexed_objects']) == len(processed_videos[metadata_id]['FiGO_scanned_objects'])):
            print(f"All objects' queries executed for video: {VIDEO_PATH}.\nFinal set of objects queried: {processed_videos[metadata_id]['TASM_indexed_objects']}")
            break
        for fgi in processed_videos[metadata_id]['FiGO_scanned_objects']:
            if(fgi not in processed_videos[metadata_id]['TASM_indexed_objects']):
                print(f"\n##### Query running for object: '{fgi}' in video '{VIDEO_PATH}'")
                os.system(f"python3 figo-tasm_video-object_query.py --video '{VIDEO_PATH}' --label '{fgi}' --out {QUERY_OUT_DIR}/ >> {RUN_LOGS_PATH}")
                query_profile_df_new = pd.read_csv(query_profile_csv_path)
                if(list(query_profile_df_new['n_objects'])[0]>0):
                    query_profile_all_df = pd.concat([query_profile_all_df, query_profile_df_new], ignore_index=True)
                elif(list(query_profile_df_new['n_objects'])[0]==0 and 0 not in list(query_profile_all_df['n_objects'])):
                    query_profile_all_df = pd.concat([query_profile_all_df, query_profile_df_new], ignore_index=True)
                else:
                    pass
                ### commit updates incase of crash
                query_profile_all_df.to_csv(query_profile_all_csv_path, index=None)
                break
    query_profile_all_df.to_csv(query_profile_all_csv_path, index=None)
    print("###### All logs saved in:",query_profile_csv_path,"\n")

### run on all object queries
# allobjectqueries('../data/birds.mp4', 'bird')
# allobjectqueries('data/v_1590672109537497088_uX7AXwvNLW-3oMdN_hevcfixed.mp4', 'car')
allobjectqueries('data/v_1592788251110428672_9AYBMr_b81UcT8ye_hevcfixed.mp4', 'truck')
allobjectqueries('data/v_1621764999487307776_9xEkuC6Qxrut_mKv_hevcfixed.mp4', 'car')
allobjectqueries('data/v_1627967517670146049_U30PJcBuyWJJmYmS_hevcfixed.mp4', 'car')
allobjectqueries('data/v_1720772891728302085_Rzs26cVD50wkjEJf_hevcfixed.mp4', 'car')

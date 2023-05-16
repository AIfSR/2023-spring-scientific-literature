import layoutparser as lp
from vila.pdftools.pdf_extractor import PDFExtractor
from vila.predictors import HierarchicalPDFPredictor, LayoutIndicatorPDFPredictor
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import pairwise_distances
from collections import defaultdict
import json
import pandas as pd
from tqdm import tqdm
from difflib import SequenceMatcher
import os
import tqdm
import pickle

class LayoutConductor():
    def __init__(self, pdf_path = './docs/', fig_path = './fig_outs/',result_directory_path='./outputs/'):
        self.load_models()
        self.pdf_path = pdf_path
        self.fig_path = fig_path
        self.result_directory_path = result_directory_path

    def load_models(self):
        self.pdf_extractor = PDFExtractor("pdfplumber")
        self.vision_model = lp.AutoLayoutModel("lp://efficientdet/PubLayNet/tf_efficientdet_d1")
        # equation_model = lp.AutoLayoutModel("lp://efficientdet/MFD/tf_efficientdet_d1")
        self.pdf_predictor = LayoutIndicatorPDFPredictor.from_pretrained("allenai/ivila-block-layoutlm-finetuned-grotoap2")
    
    def open_pdf_fig(self, fname):
        with open(self.fig_path+fname+'.json','r') as f:
            page_figs = json.load(f)
        page_tokens, page_images = self.pdf_extractor.load_tokens_and_image(self.pdf_path+fname+".pdf")

        page_fig_dict = defaultdict(list)
        for page_fig in page_figs:
            page_fig_dict[page_fig['page']].append(page_fig)
        return page_tokens, page_images, page_fig_dict
    
    def run_batch_layout_conductor_inference(self, fnames, idx_to_see=[1,2,3]):
        print('Running batch LC inference')
        for fname in fnames:
            self.run_layout_conductor_inference(fname, idx_to_see=idx_to_see)
            print('DONE pdf : ', fname)
            print('-------------------------------------------------')
            
    def run_layout_conductor_inference(self, fname, idx_to_see=None, display_images=True):
          page_tokens, page_images, page_fig_dict = self.open_pdf_fig(fname)
          image_list = []
          eq_blocks_list = []
          blocks_list = []
          if idx_to_see == None:
            range_pages = list(range(1, len(page_images)))
          else:
            range_pages = idx_to_see
            
          for idx in tqdm.tqdm(range_pages):
            page_image = page_images[idx]
            page_token = page_tokens[idx]

            blocks = self.vision_model.detect(page_image)
            # eq_blocks = equation_model.detect(page_image)
            page_token.annotate(blocks=blocks)

            pdf_data_all = page_token.to_pagedata().to_dict()

            try:
                predicted_tokens = self.pdf_predictor.predict(pdf_data_all, page_token.page_size)
                pred_tokens_all = predicted_tokens.to_dataframe()
                pred_tokens = pred_tokens_all[pred_tokens_all['type'].isin(['TABLE','FIGURE'])].reset_index()
            except Exception as e:
                continue
                pred_tokens = pd.DataFrame()

            pdf_data = page_token.tokens
            pdf_data = lp.Layout([x for x in pdf_data]).to_dataframe()[['x_1','y_1','x_2','y_2','text']]

            blocks = page_token.blocks
            blocks = lp.Layout([x for x in blocks]).to_dataframe()
            blocks['src'] = 'orig'

            ## LOAD TABLES AND FIGS AND CAPTIONS
            table_coords = []
            for page_fig in page_fig_dict[idx]:

              dict_obj = page_fig['captionBoundary']
              dict_obj['type'] = page_fig['figType'].lower()
              table_coords.append(dict_obj)

              dict_obj = page_fig['regionBoundary']
              dict_obj['type'] = page_fig['figType'].lower()
              table_coords.append(dict_obj)

            table_coords_df = pd.DataFrame(table_coords).rename(columns={'x1':'x_1','y1':'y_1','x2':'x_2','y2':'y_2'})
            table_coords_df['src'] = 'pdffig'

            ## ADD THE PDFFIG BLOCKS TO BLOCKS 
            blocks  = check_block_intersection2(table_coords_df, blocks)

            if blocks.shape[0]>=1:
                blocks['text'] = ''
                blocks['type'] = blocks['type'].str.lower()

                blocks = get_intersection_text(pdf_data, blocks)

                # compute new layout types for the blocks based on pred_tokens
                if pred_tokens.shape[0]>=1:
                  new_blocks_df, new_pred_df = update_types2(blocks,pred_tokens)
                  new_blocks_df['type'] = new_blocks_df['type'].str.lower()
                  new_pred_df = new_pred_df[new_pred_df['label'] == 0]
                  blocks = pd.concat([new_blocks_df, new_pred_df.drop(['label','index'], axis = 1)] ).reset_index()
                else:
                  blocks = blocks.reset_index()

                blocks['page_idx'] = idx
                blocks['type'] = blocks['type'].str.lower()
                blocks_list.append(blocks)
                
          ### combine all blocks from all pages, for inter-page analyses(remove headers etc)
          combined_blocks = pd.concat(blocks_list)
          if len(blocks_list)>1: # only for multi-page docs
            ## global optimzation to clean out repeat text blocks like author name/title etc.
            texts_to_block = compute_freq_labels(combined_blocks)
            # print("ttb: ",texts_to_block)
            texts_to_block = [x for x in texts_to_block if len(x)>1]

            combined_blocks = compute_sim_labels(combined_blocks.reset_index(), texts_to_block)
            combined_blocks = combined_blocks[combined_blocks['label']==0]
          
          ### Display/store the results
          
          directory_path = self.result_directory_path + fname + '_out/' 
          if not os.path.exists(directory_path):
              os.makedirs(directory_path)
          
          print('COMPUTING LAYOUT + STORING RESULTS')
          for idx in range_pages:
              img, rec_layout = compute_page_layout(page_images[idx], combined_blocks[combined_blocks['page_idx']==idx])
              if display_images==True:
                  display(img)
              img.save(directory_path + fname + "_" + str(idx) + ".jpg", 'JPEG')
              ### todo save layout
              # with open('','wb') as f:
              #     pickle.dump(rec_layout,f)

def iou(box1, box2, code = 'union'):
    x1 = max(box1['x_1'], box2['x_1'])
    y1 = max(box1['y_1'], box2['y_1'])
    x2 = min(box1['x_2'], box2['x_2'])
    y2 = min(box1['y_2'], box2['y_2'])
    intersection = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    area_box1 = (box1['x_2'] - box1['x_1'] + 1) * (box1['y_2'] - box1['y_1'] + 1)
    area_box2 = (box2['x_2'] - box2['x_1'] + 1) * (box2['y_2'] - box2['y_1'] + 1)
    if code == 'union':
      area = area_box1 + area_box2 - intersection
    elif code == 'box1':
      area = area_box1
    elif code == 'box2':
      area = area_box2
    elif code == 'min':
      area = min(area_box1, area_box2)
    iou = intersection / area
    return iou

def update_types2(df_block, df_token, thresh = 0.5, thresh_toks = 0.2):
    df_token['label'] = 0
    df_token['src'] = 'pred_tok'
    for i in range(len(df_block)):
        block_bbox = df_block.iloc[i]

        c = 0
        if block_bbox['type']!='list':
            for j in range(len(df_token)):
              if df_token.at[j, 'label'] == 0:
                token_bbox = df_token.iloc[j]
                if iou(block_bbox, token_bbox, 'min') > thresh :
                  df_token.at[j, 'label'] = 1
                  c+=1
            n_tokens = len(block_bbox['text'].split())
            if n_tokens>=1:
              if c > n_tokens*thresh_toks:
                df_block.at[i, 'type'] = 'table'
                df_block.at[i, 'src']  = 'mod_pred_toks'
    return df_block, df_token

def merge_bounding_boxes(cluster_labels, measurements):
    merged_boxes = {}
    for label, measure in zip(cluster_labels, measurements):
        if label not in merged_boxes:
            merged_boxes[label] = measure
        else:
            merged_boxes[label] = [min(x, y) if i < 2 else max(x, y) for i, (x, y) in enumerate(zip(merged_boxes[label], measure))]
    return merged_boxes

def reconstruct_layout(clustered_bbox, box_type, recon_type = 'dict'):
  if recon_type == 'dict':
    return lp.Layout([       
                  lp.TextBlock(
                      lp.Rectangle(float(ele['x1']), float(ele['y1']), 
                                  float(ele['x2']), float(ele['y2'])),
                      text=None,
                      type = box_type[idx],
                      id = idx
                  ) for idx, ele in enumerate(clustered_bbox)
              ])
  elif recon_type == 'list':
    return lp.Layout([       
                  lp.TextBlock(
                      lp.Rectangle(float(ele[0]), float(ele[1]), 
                                  float(ele[2]), float(ele[3])),
                      text=None,
                      type = box_type[idx],
                      id = idx
                  ) for idx, ele in enumerate(clustered_bbox)
              ])
  elif recon_type == 'pd':
    return lp.Layout([       
                  lp.TextBlock(
                      lp.Rectangle(float(ele[1]['x_1']), float(ele[1]['y_1']), 
                                  float(ele[1]['x_2']), float(ele[1]['y_2'])),
                      text=None,
                      type = box_type[idx],
                      id = idx
                  ) for idx, ele in enumerate(clustered_bbox.iterrows())
              ])

def check_block_intersection2(df1,df2, thresh = 0.5):
    row_to_append = []
    for i in range(len(df1)):
        bbox1 = df1.iloc[i]
        flag = True
        for j in range(len(df2)):
            bbox2 = df2.iloc[j]
            if iou(bbox1, bbox2, 'min')>thresh:
                df2.at[j, 'type'] = 'table'
                df2.at[j, 'src'] = 'modded_pdffig'
            flag = False
        if flag: 
            row_to_append.append(i)
    if len(row_to_append)>0:
        df2 = pd.concat([df2, df1.iloc[row_to_append]], ignore_index=True)
    return df2

def remove_table_intersection(df1,df2, thresh = 0.5):
  v = []
  for i in range(len(df1)):
    block_coord = {}
    bc = df1[i]
    block_coord['x_1'],block_coord['y_1'],block_coord['x_2'],block_coord['y_2'] = (bc[0],bc[1],bc[2],bc[3]) 
    flag = True
    for j in range(len(df2)):
      word_coord = {}
      wc = df2[j]
      word_coord['x_1'],word_coord['y_1'],word_coord['x_2'],word_coord['y_2'] = (wc[0],wc[1],wc[2],wc[3]) 
      if iou(block_coord, word_coord, 'min')>thresh:
        v.append(j)
  # print(v)
  for index in sorted(list(set(v)), reverse=True):
    del df2[index]
  return df2

def get_intersection_text(df1, df2, thresh = 0.5):
  # iterate through each row of df1
    for i in range(len(df1)):
        # get the coordinates of the i-th bounding box in df1
        x1_i, y1_i, x2_i, y2_i, text_i = df1.iloc[i]
        # iterate through each row of df2
        for j in range(len(df2)):
            # get the coordinates of the j-th bounding box in df2
            x1_j, y1_j, x2_j, y2_j, text_j = df2[['x_1','y_1','x_2','y_2','text']].iloc[j]
            # calculate the intersection area between the i-th and j-th bounding boxes
            intersection_area = max(0, min(x2_i, x2_j) - max(x1_i, x1_j)) * max(0, min(y2_i, y2_j) - max(y1_i, y1_j))
            # calculate the area of the i-th bounding box
            area_i = (x2_i - x1_i) * (y2_i - y1_i)
            # check if the intersection area is greater than thresh% of the area of the i-th bounding box
            if intersection_area / area_i > thresh:
                # if it is, append the text_i value to the text_j value in df2
                df2.at[j, 'text'] = df2.at[j, 'text'] + ' ' + text_i
    return df2

def similarity(s1, s2):
    if len(s1) == len(s2) ==0:
          return 0
    else:
          return SequenceMatcher(None, s1, s2).ratio() #1 - Levenshtein.distance(s1, s2) / max(len(s1), len(s2))

def compute_sim_labels(df, texts_to_block, similarity_threshold = 0.9):
  df['label'] = 0
  for i, row1 in df.iterrows():
    for j, text in enumerate(texts_to_block):
          # print(similarity(row1['text'], text), row1['text'],text)
          if  similarity(row1['text'], text)> similarity_threshold:
              df.at[i, 'label'] = 1
              break
  return df

def compute_freq_labels(df, freq_threshold = 0.5):
  # Define the minimum frequency threshold for labeling
  freq_threshold = len(df['page_idx'].unique()) * freq_threshold
  # Group the rows by their text and count the number of pages each text appears on
  freq_counts = df.groupby('text')['page_idx'].nunique()
  texts = []
  for text, freq_count in freq_counts.items():
      if freq_count >= freq_threshold:
        if len(text)>1:
            texts.append(text)
  return list(set(texts))

def compute_page_layout(page_image, pred_df):
    pred_df_tables= pred_df[pred_df['type'].isin(['table','figure'])]
    pred_df= pred_df[~pred_df['type'].isin(['table','figure'])]

    new_blocks_list = pred_df[['x_1','y_1','x_2','y_2']].values.tolist()
    new_blocks_list_x = pred_df[['x_1','x_2']].values.tolist()

    new_table_blocks_list = pred_df_tables[['x_1','y_1','x_2','y_2']].values.tolist()

    if len(new_blocks_list)>1:
        model = AgglomerativeClustering(n_clusters=None, linkage="single", metric='l1', distance_threshold=200)
        model.fit_predict(new_blocks_list_x)
        clustered_bbox_orig = list(merge_bounding_boxes(model.labels_, new_blocks_list).values())
    else:
        clustered_bbox_orig = new_blocks_list

    new_table_blocks_list = remove_table_intersection(clustered_bbox_orig, new_table_blocks_list)
    new_eq_blocks_list = []

    btype = ['bbox']*len(clustered_bbox_orig)+['tbl']*len(new_table_blocks_list) + ['eq']*len(new_eq_blocks_list)
    bboxs = clustered_bbox_orig+new_table_blocks_list+new_eq_blocks_list
    if len(bboxs) >=1:
        reconstructed_layout = reconstruct_layout(bboxs, box_type = btype, recon_type = 'list')
    else:
        reconstructed_layout = []
    return(lp.draw_box(page_image, reconstructed_layout, box_width=1,box_alpha=0.30, color_map={'bbox':'red','tbl':'blue', 'eq':'yellow'}), reconstructed_layout)
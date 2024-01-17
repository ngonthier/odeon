import os
import shutil
import streamlit as st
from PIL import Image

import geopandas as gpd
import plotly.express as px

import numpy as np
import config



st.set_page_config(layout="wide")
styl = f"""
<style>
	.block-container{{
		padding-top: 1rem;
		padding-right: 1rem;
		padding-left: 1rem;
		padding-bottom: 1rem;
	}}

    .st-emotion-cache-1r6slb0 {{
        align-self: stretch;
    }} 

</style>
"""
st.markdown(styl, unsafe_allow_html=True)
if "user" not in st.session_state:
    config.switch_page("gers_index")

st.sidebar.text(f"Bonjour {st.session_state.user}")

# print("current dir", os.getcwd())
# Define directories for input and output

if "read_only" not in st.session_state:
    st.session_state.read_only = True

empty_relabel = True
if "gpd_file" not in st.session_state:
    user_gpd_file = f"{config.post_labels_dir}/{st.session_state.stage}/gers_{st.session_state.stage}_post_labels_{st.session_state.user}.geojson"
    st.session_state.gpd_file = user_gpd_file
    #  = './notebooks/gers/validate/data.geojson'
else:
# relabeled_gpd_file = './notebooks/gers/validate/data_relabeled.geojson'
# empty_relabel = True
    if os.path.exists(st.session_state.gpd_file):
        empty_relabel = False
# dataset_rootdir = '/mnt/store_dai/datasrc/dchan/gers/change/patches'
rootdir = './notebooks'
batch_size = 1

if 'gdf' not in st.session_state:
    if empty_relabel:
        st.session_state.gdf_full = gpd.read_file(config.stages[st.session_state.stage]["gpd_file"])
        st.session_state.gdf = st.session_state.gdf_full
        st.session_state.gdf["done"] = False
        st.session_state.gdf["label_ko"] = False
        st.session_state.gdf["model_better"] = False
        st.session_state.gdf["doubt"] = False
        st.session_state.gdf["star"] = False
        st.session_state.gdf["post_comment"] = ""
        st.toast("Démarrage post-labellisation!", icon="✅")
    else:
        st.session_state.gdf_full = gpd.read_file(st.session_state.gpd_file)
        st.session_state.gdf = st.session_state.gdf_full
        st.toast("Continuation post-labellisation en cours!", icon="✅")

    st.session_state.dones = len(st.session_state.gdf[st.session_state.gdf["done"] == True])
    st.session_state.total = len(st.session_state.gdf)

    st.session_state.label_kos = len(st.session_state.gdf[st.session_state.gdf["label_ko"] == True])
    st.session_state.label_model_betters = len(st.session_state.gdf[st.session_state.gdf["model_better"] == True])
    st.session_state.doubts = len(st.session_state.gdf[st.session_state.gdf["doubt"] == True])
    st.session_state.stars = len(st.session_state.gdf[st.session_state.gdf["star"] == True])

    st.session_state.current_batch = 0
    st.session_state.only_doubts = False
    st.session_state.only_stars = False

# num_batches = len(st.session_state.gdf) // batch_size + 1

st.session_state.row = st.session_state.gdf.iloc[st.session_state.current_batch]


# Function to display images and checkboxes
def display_images_and_checkboxes(row, i):
    # Convert columns T0, T1, change, preds, logits to PIL images
    t0_image = Image.open(f"{rootdir}/{row['T0']}")
    t1_image = Image.open(f"{rootdir}/{row['T1']}")
    change_image = Image.open(f"{rootdir}/{row['change']}")
    preds_image = Image.open(f"{rootdir}/{row['preds']}")
    logits_image = Image.open(f"{rootdir}/{row['logits']}")

    with st.container():
        col21, col22, col23, col24 = st.columns(4)
        last = col21.button("<", use_container_width=True, on_click=on_last)
        save = col22.button("Save", use_container_width=True, on_click=on_save, disabled=st.session_state.read_only)
        goto_last = col23.button("Go To First Undone", use_container_width=True, on_click=on_goto_last)
        next = col24.button("\\>", use_container_width=True, on_click=on_next)
        
        col2, col3 = st.columns(2)

        col3.toggle('Post-Label Done', value = st.session_state.gdf.at[row.name, 'done'], on_change=on_done)

        deflts = []
        if st.session_state.gdf.at[row.name, 'label_ko']:
            deflts.append("Label KO")
        if st.session_state.gdf.at[row.name, 'model_better']:
            deflts.append("Model Better")
        if st.session_state.gdf.at[row.name, 'doubt']:
            deflts.append("Doubt")
        if st.session_state.gdf.at[row.name, 'star']:
            deflts.append("Star")            
        relabel = col2.multiselect(
            'Relabel', label_visibility="hidden", options= ["Label KO", "Model Better", "Doubt", "Star"], key = f"relabel_{i}",
            default = deflts
        )

        arr = np.vstack((
            np.expand_dims(np.array(t0_image), axis=0),
            np.expand_dims(np.array(t1_image), axis=0),
            np.expand_dims(np.array(change_image), axis=0),
            np.expand_dims(np.array(preds_image), axis=0),
            # np.expand_dims(np.array(logits_image), axis=0),
        ))
    
        fig = px.imshow(arr, binary_string=True, 
                        facet_col=0, facet_col_wrap=4, facet_col_spacing=0 ,facet_row_spacing=0)

        fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
        fig.update_layout(
            margin=dict(l=1,r=1,b=0,t=20, pad=0),
            # yaxis={'visible': False, 'showticklabels': False},
            # xaxis={'visible': False, 'showticklabels': False}
        )        
        # fig.layout.annotations[0]['text'] = 'Logits'
        fig.layout.annotations[0]['text'] = 'T0'
        fig.layout.annotations[1]['text'] = 'T1'
        fig.layout.annotations[2]['text'] = 'Change'
        fig.layout.annotations[3]['text'] = 'Preds'

            
        st.plotly_chart(fig, use_container_width=True)
        
        with st.expander("post_comment + logits"):
            col31, col32 = st.columns(2)
            col31.text_area(label="Post-Comment", value = st.session_state.gdf.at[row.name, 'post_comment'], height=10, placeholder="Optional Post-Label Comment", label_visibility="hidden")
            fig = px.imshow(np.array(logits_image), title="Logits")
            fig.update_xaxes(showticklabels=False).update_yaxes(showticklabels=False)
            fig.update_layout(
                margin=dict(l=1,r=1,b=0,t=20, pad=0),
                # yaxis={'visible': False, 'showticklabels': False},
                # xaxis={'visible': False, 'showticklabels': False}
            )        
            # fig.layout.annotations['text'] = 'Logits'
            col32.plotly_chart(fig, use_container_width=True)

    # Save checkbox states in new columns of GeoDataFrame
    if "Label KO" in relabel:
        st.session_state.gdf.at[row.name, 'label_ko'] = True
    else:
        st.session_state.gdf.at[row.name, 'label_ko'] = False

    if "Model Better" in relabel:
        st.session_state.gdf.at[row.name, 'model_better'] = True
    else:
        st.session_state.gdf.at[row.name, 'model_better'] = False

    if "Doubt" in relabel:
        st.session_state.gdf.at[row.name, 'doubt'] = True
    else:
        st.session_state.gdf.at[row.name, 'doubt'] = False

    if "Star" in relabel:
        st.session_state.gdf.at[row.name, 'star'] = True
    else:
        st.session_state.gdf.at[row.name, 'star'] = False

# Initialize batch parameters

# Main Streamlit app
if not st.session_state.read_only:
    st.markdown(f"## Dataset Gers {st.session_state.stage} Post-Labellisation")
else:
    st.markdown(f"## Dataset Gers {st.session_state.stage} Post-Labellisation (LECTURE SEULE)")

def slide():
    st.session_state.current_batch = st.session_state.my_slider

def on_next():
    st.session_state.current_batch = min(len(st.session_state.gdf), st.session_state.current_batch + 1)
    st.session_state.my_slider = st.session_state.current_batch

def on_last():
    st.session_state.current_batch = max(0, st.session_state.current_batch - 1)
    st.session_state.my_slider = st.session_state.current_batch
    
def on_save():
    st.session_state.gdf.to_file(st.session_state.gpd_file, driver="GeoJSON")
    st.toast("GeoDataFrame saved successfully!")

def on_goto_last():
    b = st.session_state.gdf.done.ne(True).idxmax()
    st.session_state.current_batch = max(0, b)
    st.session_state.my_slider = st.session_state.current_batch

def on_done():
    st.session_state.gdf.at[st.session_state.row.name, 'done'] = not st.session_state.gdf.at[st.session_state.row.name, 'done'] 
    # if st.session_state.gdf.at[st.session_state.row.name, 'done']:
    #     st.session_state.dones += 1
    # else:
    #     st.session_state.dones -= 1

    st.session_state.dones = len(st.session_state.gdf[st.session_state.gdf["done"] == True])
    st.session_state.label_kos = len(st.session_state.gdf[st.session_state.gdf["label_ko"] == True])
    st.session_state.label_model_betters = len(st.session_state.gdf[st.session_state.gdf["model_better"] == True])
    st.session_state.doubts = len(st.session_state.gdf[st.session_state.gdf["doubt"] == True])
    st.session_state.stars = len(st.session_state.gdf[st.session_state.gdf["star"] == True])

def on_only_doubts():
    st.session_state.only_doubts = not st.session_state.only_doubts
    if st.session_state.only_doubts:
        st.session_state.gdf = st.session_state.gdf_full[st.session_state.gdf_full["doubt"] == True]
        st.session_state.current_batch = 0
    else:
        st.session_state.gdf = st.session_state.gdf_full
    st.session_state.dones = len(st.session_state.gdf[st.session_state.gdf["done"] == True])
    st.session_state.label_kos = len(st.session_state.gdf[st.session_state.gdf["label_ko"] == True])
    st.session_state.label_model_betters = len(st.session_state.gdf[st.session_state.gdf["model_better"] == True])
    st.session_state.doubts = len(st.session_state.gdf[st.session_state.gdf["doubt"] == True])
    st.session_state.stars = len(st.session_state.gdf[st.session_state.gdf["star"] == True])
    st.session_state.total = st.session_state.dones
    
    
def on_only_stars():
    st.session_state.only_stars = not st.session_state.only_stars
    if st.session_state.only_stars:
        st.session_state.gdf = st.session_state.gdf_full[st.session_state.gdf_full["star"] == True]
        st.session_state.current_batch = 0
    else:
        st.session_state.gdf = st.session_state.gdf_full
    st.session_state.dones = len(st.session_state.gdf[st.session_state.gdf["done"] == True])
    st.session_state.label_kos = len(st.session_state.gdf[st.session_state.gdf["label_ko"] == True])
    st.session_state.label_model_betters = len(st.session_state.gdf[st.session_state.gdf["model_better"] == True])
    st.session_state.doubts = len(st.session_state.gdf[st.session_state.gdf["doubt"] == True])
    st.session_state.stars = len(st.session_state.gdf[st.session_state.gdf["star"] == True])
    st.session_state.total = st.session_state.dones
    
col01, col02 = st.columns([1.0, 9.0])
col01.markdown(f'''Batch {st.session_state.current_batch}''')
col02.markdown(f'''(Relabeled {(st.session_state.dones / st.session_state.total) * 100:.2f}% - {st.session_state.dones}/{st.session_state.total} - label_ko {st.session_state.label_kos} / model_better {st.session_state.label_model_betters} / doubt {st.session_state.doubts} / star {st.session_state.stars})''')

col1, col2 = st.columns([9.0, 1.0])
batch = col1.slider(
    "Batch Number",
    label_visibility="hidden",
    min_value=0,
    max_value=st.session_state.total,
    value=0,
    step = 1,
    key="my_slider",
    on_change=slide
)

col2.toggle('Only Doubts', value = st.session_state.only_doubts, on_change=on_only_doubts)
col2.toggle('Only stars', value = st.session_state.only_stars, on_change=on_only_stars)

display_images_and_checkboxes(st.session_state.row, st.session_state.current_batch)

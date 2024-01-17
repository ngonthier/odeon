import os
import shutil
import streamlit as st
from PIL import Image

import geopandas as gpd
import plotly.express as px

import numpy as np
import config
import glob


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
    st.markdown("""
        # Dataset Gers Post-Labellisation
    """, unsafe_allow_html=True)

    user = st.text_input("Entrez votre nom d'utilisateur:")
    if user:
        st.session_state.user = user
        st.rerun()

else:

    st.markdown(f"""
        # Dataset Gers Post-Labellisation (Bonjour {st.session_state.user})
    """, unsafe_allow_html=True)

    st.sidebar.text(f"Bonjour {st.session_state.user}")
    for stage, v in config.stages.items():
        st.markdown(f"""
    ## Dataset Gers {stage}
""", unsafe_allow_html=True)
        user_gpd_file = f"{config.post_labels_dir}/{stage}/gers_{stage}_post_labels_{st.session_state.user}.geojson"
        if os.path.exists(user_gpd_file):
            if st.button(f"**Continuer Votre Post-Labellisation {stage} en cours** ({user_gpd_file})", type="primary"):
                st.session_state.stage = stage
                st.session_state.gpd_file = user_gpd_file
                st.session_state.read_only = False
                config.switch_page("gers_relabel")
        else:
            if st.button(f"**Démarrer Une Nouvelle Post-Labellisation {stage}**", type="primary"):
                st.session_state.stage = stage
                st.session_state.read_only = False
                config.switch_page("gers_relabel")

        files = [f for f in glob.glob(f"{config.post_labels_dir}/{stage}/*.geojson")]
        st.markdown(f'#### Voir autre Post-labellisations')
        st.session_state.sel = st.selectbox(
            f'Post-labellisations {stage} existantes',
            files,
        )
        if st.button(f"Voir cette post-labellisation {stage}"):
            st.session_state.stage = stage
            st.session_state.gpd_file = st.session_state.sel
            st.session_state.read_only = True            
            config.switch_page("gers_relabel")

        st.markdown("----")

    # option = st.selectbox(
    #     'Selectionnez une phase à post-labélliser',
    # ('Email', 'Home phone', 'Mobile phone'))
    # if st.button("GO TO Post-Label"):
    #     switch_page("gers_relabel")

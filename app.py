#app.py

import aaa
import maladie
import detection_de_zone_irriguer
import streamlit as st

PAGES = {
    "base de donne": aaa,
    " detection des maladie des plantes": maladie,
    'detection de zone irriguer': detection_de_zone_irriguer
}
st.sidebar.title('Navigation')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()

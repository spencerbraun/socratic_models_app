# Socratic Models App

A small app meant to test the capabilities explored in the [Socratic Models](https://socraticmodels.github.io/) paper. The idea is to use a committee of large language models (LLM), video-language models (VLM), e.g. CLIP, audio-language models (ALM) like wav2clip, and speech to text models to supervise each other. This demo explores the interaction between LLMs and VLMs using YouTube videos through a Streamlit app.

## Setup
To run the app, a valid HuggingFace API key is needed. Create a key at [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and save to a file `src/hf_api.key`.

## Run
Navigate to the `src` dir and run `streamlit run app.py`.
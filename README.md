# probe-across-time


**Released!** Feel free to checkout our model [here](https://drive.google.com/drive/folders/1i7cNInCmiW07m_mfAmD3s4ZxcgJqiH6Y?usp=sharing). From our point of view, researchers might be more familiar with using `huggingface`, so those checkpoints were in huggingface format.

### Notes:
* As an illustrative example, folder `NEWS(12GB)` denotes the checkpoints from pretraining RoBERTa-base on downsampled RealNews, see more details in the paper.
* Within folders like, `NEWS(12GB)`, you will see `roberta_bz256_savesteps` or `roberta_bz256_saveepochs` or both; those names denote how the checkpoints (within those folders) are saved during prertaining **by epoch or step**. The name of a checkpoint folder is `checkpoint-xx`


# Scripts included in this repo
* `convert_pytorch_to_roberta_original_pytorch_checkpoint.py`: we realize some probe like LAMA is most convenient to be run on `fairseq` model, so we provide this converter script.

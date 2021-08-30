# Probing across time

**Released!** Feel free to checkout our model [here](https://arkdata.cs.washington.edu/probe-across-time/). From our point of view, researchers might be more familiar with using `huggingface`, so those checkpoints were in huggingface format.

## Batched Download
In each files under directory `checkpoint_paths`, we provide the paths to all `.tar` files. One could modify the files accordingly and download with commands like one below (For more options, see [here](https://stackoverflow.com/questions/13939038/how-do-you-run-a-command-for-each-line-of-a-file)).

```bash
xargs -n 1 -I{} wget https://arkdata.cs.washington.edu/probe-across-time/{} --no-check-certificate < <(cat path/to/modified/file)

```

### Notes:
* As an illustrative example, folder `NEWS` denotes the checkpoints from pretraining RoBERTa-base on downsampled RealNews, see more details in the paper.
* Within folders like, `NEWS`, you will see `roberta_bz256_savesteps` or `roberta_bz256_saveepochs` or both; those names denote how the checkpoints (within those folders) are saved during prertaining **by epoch or step**. The name of a checkpoint folder is `checkpoint-xx`


# Scripts included in this repo
* `convert_pytorch_to_roberta_original_pytorch_checkpoint.py`: we realize some probe like LAMA is most convenient to be run on `fairseq` model, so we provide this converter script.

from argparse import ArgumentParser


def get_parser():
    parser = ArgumentParser()

    parser.add_argument("--model", type=str, default="Attention")

    # Hyperparameters for the model
    parser.add_argument("--hidden_dim", type=int, default=[512], nargs="+")
    parser.add_argument("--attention_dim", type=int, default=[256], nargs="+")

    # Trainer arguments
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight_decay", type=float, default=10e-4)
    parser.add_argument("--max_epochs", type=int, default=100)

    # Dataset arguments
    parser.add_argument("--patch_size", type=int, default=299)
    parser.add_argument("--patches_per_pat", type=int, default=10)
    parser.add_argument("--min_patches_per_pat", type=int, default=100)
    parser.add_argument("--training_strategy", type=str, default='random_tiles')
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--cv_split_path", type=str, default='/home/p163v/histopathology/splits/02022024/')
    parser.add_argument("--metadata_column", type=str, default=None)
    parser.add_argument("--embedding", type=str, default='retccl')
    parser.add_argument("--tissue_filter", type=str, default=None, nargs="+")
    parser.add_argument("--label_path", type=str, default="/home/p163v/histopathology/metadata/CT_3_Class_Draft.csv")
    parser.add_argument("--slide_annot_path", type=str, default="/home/p163v/histopathology/metadata/labels_with_new_batch.csv")
    
    # Transformer Specific Arguments
    parser.add_argument("--n_heads", type=int, default=None)
    parser.add_argument("--clip_grad", type=float, default=0.0)
    # parser.add_argument("--swa", action='store_true')

    # parser.add_argument("--oncotree_filter", type=str, default=None, nargs="+")
    # parser.add_argument("--classifier_confidence", type=int, default=80)

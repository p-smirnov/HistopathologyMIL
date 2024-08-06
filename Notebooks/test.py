import argparse
parser = argparse.ArgumentParser(description='Extract embeddings from UNI model')
parser.add_argument('--slide_names', type=str, default=['0A2F095A-1117-4F72-B648-717BAA3FD4AE'], help='Slide name', nargs="+")
args = parser.parse_args()
print(args.slide_names)

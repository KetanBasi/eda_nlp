# Easy data augmentation techniques for text classification
# Jason Wei and Kai Zou

# arguments to be parsed from command line
import argparse

# import random

from eda import eda
from os.path import dirname, basename, join
from typing import Union

ap = argparse.ArgumentParser()
ap.add_argument(
    "-i",
    "--input",
    required=True,
    type=str,
    help="input file of unaugmented data",
)
ap.add_argument(
    "-o",
    "--output",
    required=False,
    type=str,
    help="output file of augmented data",
)
ap.add_argument(
    "-l",
    "--language",
    required=False,
    type=str,
    help="language of the input file",
    default="eng",
)
ap.add_argument(
    "-s",
    "--separator",
    required=False,
    type=str,
    help='column separator of the input file: "\\t", ",", ";" etc.',
    default="\t",
)
ap.add_argument(
    "--skiprows",
    required=False,
    type=int,
    help="number of rows to skip in the input file",
    default=0,
)
ap.add_argument(
    "-g",
    "--gen_only",
    required=False,
    action="store_true",
)
ap.add_argument(
    "-n",
    "--num_aug",
    required=False,
    type=float,
    help=(
        "number of augmented sentences per original sentence (int). Percentages (0 < n < 1) "
        "also allowed but there's inconsistency issue with the number of augmented sentences "
        "generated.\n"
    ),
    default=9,
)
ap.add_argument(
    "--alpha_sr",
    required=False,
    type=float,
    help="percent of words in each sentence to be replaced by synonyms (0 <= alpha_sr <= 1)",
    default=0.1,
)
ap.add_argument(
    "--alpha_ri",
    required=False,
    type=float,
    help="percent of words in each sentence to be inserted (0 <= alpha_ri <= 1)",
    default=0.1,
)
ap.add_argument(
    "--alpha_rs",
    required=False,
    type=float,
    help="percent of words in each sentence to be swapped (0 <= alpha_rs <= 1)",
    default=0.1,
)
ap.add_argument(
    "--alpha_rd",
    required=False,
    type=float,
    help="percent of words in each sentence to be deleted (0 <= alpha_rd <= 1)",
    default=0.1,
)
args = ap.parse_args()

# the output file
if args.output:
    output = args.output
else:
    output = join(dirname(args.input), "eda_" + basename(args.input))

if args.alpha_sr == args.alpha_ri == args.alpha_rs == args.alpha_rd == 0:
    ap.error("At least one alpha should be greater than zero")


# generate more data with standard augmentation
def gen_eda(
    train_orig: str,
    output_file: str,
    language: str,
    separator: str,
    alpha_sr: float,
    alpha_ri: float,
    alpha_rs: float,
    alpha_rd: float,
    num_aug: Union[int, float] = 9,
    skip_lines: int = 0,
    gen_only: bool = False,
):
    """
    Generate more data with standard augmentation.

    Args:
        train_orig (str): path to the original training file.
        output_file (str): path to the output file.
        language (str): language of the input file e.g. eng, deu, ind, etc.
        separator (str): column separator of the input file: "\t", ",", ";" etc.
        alpha_sr (float): percent of words in each sentence to be replaced by synonyms (0 <= alpha_sr <= 1).
        alpha_ri (float): percent of words in each sentence to be inserted (0 <= alpha_sr <= 1).
        alpha_rs (float): percent of words in each sentence to be swapped (0 <= alpha_sr <= 1).
        alpha_rd (float): percent of words in each sentence to be deleted (0 <= alpha_sr <= 1).
        num_aug (int or float, optional): number of augmented sentences per original sentence. Defaults to 9.
        skip_lines (int, optional): number of rows to skip in the input file. Defaults to 0.
        gen_only (bool, optional): if True, only return the generated sentences. Defaults to False.
    """
    with open(output_file, "w") as file_out, open(train_orig, "r") as file_source:
        # ? Skip header if any and just copy it to output file
        for _ in range(skip_lines):
            file_out.write(next(file_source))

        # ? Augmenting sentences and writing to output file
        for i, line in enumerate(file_source):
            parts = line[:-1].split(separator)
            label = parts[0]
            sentence = parts[1]
            aug_sentences = eda(
                sentence,
                language=language,
                alpha_sr=alpha_sr,
                alpha_ri=alpha_ri,
                alpha_rs=alpha_rs,
                alpha_rd=alpha_rd,
                num_aug=num_aug,
                gen_only=gen_only,
            )
            file_buf = "".join(
                label + separator + aug_sentence + "\n"
                for aug_sentence in aug_sentences
            )
            file_out.write(file_buf)

        # ? Summary of the task
        print(
            f"generated augmented sentences with eda for {train_orig} "
            f"to {output_file} with num_aug={num_aug}, alpha_sr={alpha_sr}, "
            f"alpha_ri={alpha_ri}, alpha_rs={alpha_rs}, alpha_rd={alpha_rd}"
        )


# main function
if __name__ == "__main__":
    try:
        # generate augmented sentences and output into a new file
        gen_eda(
            args.input,
            output,
            language=args.language,
            separator=args.separator,
            alpha_sr=args.alpha_sr,
            alpha_ri=args.alpha_ri,
            alpha_rs=args.alpha_rs,
            alpha_rd=args.alpha_rd,
            num_aug=args.num_aug,
            skip_lines=args.skiprows,
            gen_only=args.gen_only,
        )
    except KeyboardInterrupt:
        print("\nKeyboardInterrupt")
        pass

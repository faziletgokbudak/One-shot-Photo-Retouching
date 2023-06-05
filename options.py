import argparse


class Options():

    def initialize(self, parser):
        parser.add_argument('--input_path', type=str)
        parser.add_argument('--output_path', type=str)
        parser.add_argument('--test_path', type=str)
        parser.add_argument('--test_output_path', type=str)
        parser.add_argument('--model_path', type=str,
                            default='/Users/faziletgokbudak/One-shot-Photo-Retouching/models/ours_room_BF')

        parser.add_argument('--lr', type=int, default=1e-2)
        parser.add_argument('--num_channel', type=int, default=3)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--epoch', type=int, default=400)

        parser.add_argument('--patch_size', type=list, default=[3, 3])
        parser.add_argument('--laplacian_level', type=int, default=5)
        parser.add_argument('--num_mlp', type=int, default=1) # number of MLPs to train together
        parser.add_argument('--num_matrices', type=int, default=256) # number of matrices to train together
        parser.add_argument("--num_matrices_list", nargs="+", default=[1, 4, 8, 16, 64, 256])

        parser.add_argument('--freq_num', type=int, default=0)
        parser.add_argument('--chrom', type=str, default=False) #trains/tests chrominance channels as well
        parser.add_argument('--var_loss', type=bool, default=False)# adds variance loss to l1 loss

        parser.add_argument('--resize_size', type=int, default=100)
        parser.add_argument('--test_image', type=str, default='input')

        return parser

    def gather_options(self):
        # initialize parser with basic options
        parser = argparse.ArgumentParser(
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser = self.initialize(parser)

        opt = parser.parse_args()
        return opt

    def parse(self, save=False):
        self.opt = self.gather_options()
        return self.opt
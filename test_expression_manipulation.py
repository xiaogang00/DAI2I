from run_expression_manipulation import *
import torch
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import os


class Engine_test(Engine):
    def parse_args_test(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        parser.add_argument('mode')
        parser.add_argument('--model_path', required=True)
        parser.add_argument('-gpu', default='0', help='the gpu index to run')
        parser.add_argument('-sp', '--save_dir', default='test/v0')
        parser.add_argument('--data_dir', help='dir of the testing iamges')
        # the options for writing
        parser.add_argument('--store_type', choices=['cat', 'sep', 'sep2'], nargs='+', default=['cat'], 
                            help='if cat, then store all transformed images in a row; if sep, then store them separately in an folder')
        # the options of testing 
        parser.add_argument('--test_list', default=None, help='a txt file of testinig list. if None, then use all images in the data-dir')
        parser.add_argument('--crop_size', default=None, type=int, help='if none, then use config, if 0, then do not crop')

        return parser

    def basic_setting(self):
        os.environ['CUDA_VISIBLE_DEVICES'] = self.args.gpu

    def label2onehot(self, labels, dim):
        """Convert label indices to one-hot vectors."""
        batch_size = labels.size(0)
        out = torch.zeros(batch_size, dim)
        out[np.arange(batch_size), labels.long()] = 1
        return out

    def load_dataset(self):
        from data import datasetSimple
        import glob
        if self.args.test_list is None:
            dst_list = glob.glob(self.args.data_dir + '/*.jpg')
        else:
            with open(self.args.test_list, 'r') as f:
                dst_list = ['{}/{}'.format(self.args.data_dir, tmp.rstrip()) for tmp in f]
        if self.args.crop_size is None:
            crop_size = (self.config['dst_crop_size'], self.config['dst_crop_size'])
        else:
            crop_size = (self.args.crop_size, self.args.crop_size)
        dst_dataset = datasetSimple.Dataset(image_list=dst_list, 
                                            crop_size=crop_size,
                                            return_filename=True)
        return dst_dataset

    def test(self):
        model = self.define_model()
        model = [tmp.cuda() for tmp in model]
        basemodel, G, _, _ = model
        G.load_state_dict(torch.load('{}/G-latest.pth'.format(self.args.model_path)))
        G.eval()

        dataset = self.load_dataset()
        loader = DataLoader(dataset, batch_size=1)
        n_branches = len(self.config['attr_names'])

        with torch.no_grad():
            for data in tqdm(loader):
                img, name = data
                img = util.toVariable(img).cuda()
                im_out = [img]
                img_BA = G(img, 0, img)
                for i in range(n_branches):
                    c_trg = self.label2onehot(torch.ones(img.size(0)) * i, n_branches).cuda()
                    img_BAG = basemodel(img_BA, c_trg)
                    img_BAGB = G(img_BAG, 1, img)
                    im_out += [img_BAGB]
                self._store_images(im_out, name)

    def _store_images(self, im_out, name):
        for store_type in self.args.store_type:
            store_dir = '{}/{}-{}'.format(self.args.model_path, self.args.save_dir, store_type)
            util.mkdir(store_dir)
            if store_type == 'cat':  # catenate the results and store them in one image for easily visualization
                im_out_cat = torch.cat(im_out)
                im_out_cat = (im_out_cat + 1) / 2 * 255
                save_path_now = '{}/{}'.format(store_dir, name[0])
                visualization.writeTensor(save_path_now, im_out_cat.data.cpu(), nRow=1)
                os.chmod(save_path_now,0o007)
            elif store_type == 'sep':  # store the results separately. each input image is one folder
                save_path_now = '{}/{}'.format(store_dir, name[0])
                util.mkdir(save_path_now)
                for i, im_out_ in enumerate(im_out):
                    im_out_ = (im_out_ + 1) / 2 * 255 
                    visualization.writeTensor('{}/{}.jpg'.format(save_path_now, i), im_out_.data.cpu(), nRow=1)
            elif store_type == 'sep2':  # store the results separately. each branch is one folder
                for i, im_out_ in enumerate(im_out):
                    save_path_now = '{}/{}'.format(store_dir, i)
                    util.mkdir(save_path_now)
                    im_out_ = (im_out_ + 1) / 2 * 255 
                    visualization.writeTensor('{}/{}.jpg'.format(save_path_now, name[0]), im_out_.data.cpu(), nRow=1)
            else:
                raise NotImplementedError


    def run(self):
        parser = self.parse_args_test()
        self.args = parser.parse_args()
        self.config_this = OrderedDict()
        self.args.config_path = '{}/options.yaml'.format(self.args.model_path)
        with open(self.args.config_path, 'r') as f:
            self.config = yaml.load(f)
        
        self.basic_setting()
        self.test()

if __name__ == '__main__':
    engine = Engine_test()
    engine.run()

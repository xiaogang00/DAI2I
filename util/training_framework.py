from __future__ import print_function
from torch.utils.data import DataLoader
from .logger import logger
from tqdm import tqdm
log = logger(True)
from .opt import opt


class TrainEngine(object):
    def __init__(self, dataset, optimizer, batch_size, data_adapter=None, recover_step_epoch=True, config=opt()):
        super(TrainEngine, self).__init__()
        self.dataset = dataset
        self.config = opt()
        self.config.num_workers = 4
        self.config.n_heavy = 5
        self.config.n_show_loss = 100
        self.config.n_save_img = 100
        self.config.n_save_model = 200
        self.config.merge_dict(config)
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=self.config.num_workers,
                                     drop_last=True)
        self.data_adapter = data_adapter
        self.optimizer = optimizer
        self.recover_step_epoch = recover_step_epoch
        with open(optimizer.opt.save_dir + '/options.yaml', 'w') as f:
            print(optimizer.opt, file=f)
            print(optimizer.opt)

    def run(self, epoch):
        global_step = 0
        e0 = 0
        global_step = self.optimizer.global_step
        e0 = self.optimizer.epoch
        for e in range(e0, epoch):
            for iter, data in enumerate(tqdm(self.dataloader)):
                if self.data_adapter is not None:
                    data = self.data_adapter(data)
                self.optimizer.set_input(data)

                self.optimizer.optimize_parameters(global_step)
                if global_step % self.config.n_show_loss == 0:
                    tqdm.write(self.optimizer.print_current_errors(e, iter, record_file=self.optimizer.opt.save_dir,
                                                                   print_msg=False))
                    self.optimizer.add_summary(global_step)
                if (global_step) % self.config.n_save_img == 0:
                    log('save samples ', global_step)
                    self.optimizer.save_samples(global_step)
                if global_step > 0 and global_step % self.config.n_save_model == 0:
                    self.optimizer.global_step = global_step
                    self.optimizer.epoch = e
                    self.optimizer.save()
                global_step += 1
        self.optimizer.save()


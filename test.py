import argparse, time, os
import imageio
from multiprocessing import Process, Queue
import options.options as option
from utils import util
from solvers import create_solver
from data import create_dataloader
from data import create_dataset
import numpy as np


class Paralle_save_img():
    def __init__(self, n_processes=4):
        self.n_processes = n_processes
        self.queue = Queue()

    def begin_background(self):
        def bg_target(queue):
            while True:
                if not queue.empty():
                    filename, img = queue.get()
                    if filename is None: break
                    # imageio.imwrite(filename, img)
                    np.save(filename, img)
        self.process = [
            Process(target=bg_target, args=(self.queue,))
            for _ in range(self.n_processes)
        ]

        for p in self.process:
            p.start()

    def end_background(self):
        for _ in range(self.n_processes):
            self.queue.put((None, None))
        while not self.queue.empty():
            time.sleep(1)
        for p in self.process:
            p.join()

    def put_image_path(self, filename, img):
        self.queue.put((filename, img))


def main():
    args = option.add_args()
    opt = option.parse(args)
    opt = option.dict_to_nonedict(opt)

    # initial configure
    scale = opt['scale']
    degrad = opt['degradation']
    network_opt = opt['networks']
    model_name = network_opt['which_model'].upper()
    if opt['self_ensemble']:
        model_name += 'plus'

    # create test dataloader
    bm_names =[]
    test_loaders = []
    for _, dataset_opt in sorted(opt['datasets'].items()):
        test_set = create_dataset(dataset_opt)
        test_loader = create_dataloader(test_set, dataset_opt)
        test_loaders.append(test_loader)
        print('===> Test Dataset: [%s]   Number of images: [%d]' % (test_set.name(), len(test_set)))
        bm_names.append(test_set.name())

    # create solver (and load model)
    solver = create_solver(opt)
    # Test phase
    print('===> Start Test')
    print("==================================================")
    print("Method: %s || Scale: %d || Degradation: %s"%(model_name, scale, degrad))

    # whether save the SR image?
    if opt['save_image']:
        para_save = Paralle_save_img()
        para_save.begin_background()
    # with para_save.begin_background() as para_save_imag

    for bm, test_loader in zip(bm_names, test_loaders):
        print("Test set : [%s]" % bm)

        total_psnr = []
        total_ssim = []
        total_time = []

        need_HR = False if test_loader.dataset.__class__.__name__.find('LRHR') < 0 else True

        if need_HR:
            save_img_path = os.path.join('./results/SR/' + degrad, model_name, bm, "x%d" % scale)
        else:
            save_img_path = os.path.join('./results/SR/' + degrad, model_name, bm, "x%d" % scale)

        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)

        for iter, batch in enumerate(test_loader):
            solver.feed_data(batch, need_HR=need_HR)

            # calculate forward time
            t0 = time.time()
            solver.test()
            t1 = time.time()
            total_time.append((t1 - t0))

            visuals = solver.get_current_visual(need_HR=need_HR)

            # calculate PSNR/SSIM metrics on Python
            if need_HR:
                psnr, ssim = util.pan_calc_metrics(visuals['SR'], visuals['HR'], opt['img_range'], crop_border=scale)
                total_psnr.append(psnr)
                total_ssim.append(ssim)
                print("[%d/%d] %s || CC(dB)/RMSE: %.4f/%.4f || Timer: %.4f sec ." % (iter + 1, len(test_loader),
                                                                                       os.path.basename(
                                                                                           batch['LR_path'][0]),
                                                                                       psnr, ssim,
                                                                                       (t1 - t0)))
            else:
                print("[%d/%d] %s || Timer: %.4f sec ." % (iter + 1, len(test_loader),
                                                           os.path.basename(batch['LR_path'][0]),
                                                           (t1 - t0)))
            if opt['save_image']:
                name = ('x{}_' + model_name + '_' ).format(scale) + os.path.basename(batch['LR_path'][0])

                para_save.put_image_path(filename=os.path.join(save_img_path, name), img=visuals['SR'])

        if need_HR:
            print("---- Average PSNR(dB) /SSIM /Speed(s) for [%s] ----" % bm)
            print("CC: %.4f      RMSE: %.4f      Speed: %.4f" % (sum(total_psnr) / len(total_psnr),
                                                                   sum(total_ssim) / len(total_ssim),
                                                                 (sum(total_time)-total_time[0]) / (len(total_time)-1)))
        else:
            print("---- Average Speed(s) for [%s] is %.4f sec ----" % (bm,

                                                                       (sum(total_time)-total_time[0]) / (len(total_time)-1)))
    if opt['save_image']:
        para_save.end_background()

    print("==================================================")
    print("===> Finished !")


if __name__ == '__main__':
    main()

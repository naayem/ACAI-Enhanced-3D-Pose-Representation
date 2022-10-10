import pprint

from deep_cvlab import utils, core

def main():

    ### initialize Trainer
    cfg = utils.utils.parse_args().cfg
    trainer = core.trainer.Trainer(cfg)

    ### copy yaml description file to the save folder
    utils.utils.copy_exp_file(trainer)

    ### copy proc.py file to the save folder
    utils.utils.copy_proc_file(trainer)

    trainer.logger.info(pprint.pformat(trainer.cfg))
    trainer.logger.info('#'*100)

    ### run the training procedure
    trainer.run()

    print('##################DONE############################')


if __name__ == '__main__':

    main()


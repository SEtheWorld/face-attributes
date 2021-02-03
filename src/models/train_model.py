from pathlib import Path
import multiprocessing
import pandas as pd
import hydra
from hydra.utils import to_absolute_path
import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, ModelCheckpoint
from factory import get_model, get_scheduler
from generator import ImageSequence
import argparse

parse = argparse.ArgumentParser()
parse.add_argument('--cfg', required=True, help='Choose the config file to train model')
args = parse.parse_args()


@hydra.main(config_path=args['cfg'])
def main(cfg):
    if cfg.wandb.project:
        import wandb
        from wandb.keras import WandbCallback
        wandb.init(project=cfg.wandb.project)
        callbacks = [WandbCallback()]
    else:
        callbacks = []

    train_csv_path = Path(to_absolute_path(__file__)).parent.joinpath("/home/Data/all", f"{cfg.data.train}.csv")
    val_csv_path = Path(to_absolute_path(__file__)).parent.joinpath("/home/Data/all", f"{cfg.data.val}.csv")

    train_df = pd.read_csv(str(train_csv_path))
    val_df = pd.read_csv(str(val_csv_path))
    train_gen = ImageSequence(cfg, train_df, "train")
    val_gen = ImageSequence(cfg, val_df, "val")

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = get_model(cfg)
        scheduler = get_scheduler(cfg)

    checkpoint_dir = Path(to_absolute_path(__file__)).parent.joinpath("checkpoint")
    checkpoint_dir.mkdir(exist_ok=True)
    filename = "_".join([cfg.model.model_name,
                         str(cfg.model.img_size),
                         "weights.{epoch:02d}-{val_loss:.2f}.hdf5"])
    callbacks.extend([
        LearningRateScheduler(schedule=scheduler),
        ModelCheckpoint(str(checkpoint_dir) + "/" + filename,
                        monitor="val_loss",
                        verbose=1,
                        save_best_only=True,
                        mode="auto")
    ])

    history = model.fit(train_gen, epochs=cfg.train.epochs, callbacks=callbacks, validation_data=val_gen,
              workers=multiprocessing.cpu_count())
   
    # with strategy.scope():
    #     model = get_model(cfg)
    #     opt = get_optimizer(cfg)
    #     loss = get_loss(cfg.loss.age, cfg.loss.gender)
    #     scheduler = get_scheduler(cfg)
    #     model.compile(optimizer=opt,
    #                   loss={loss[0], loss[1]},
    #                   metrics={"mae", "accuracy"},
    #                   loss_weights = {1, 10}
    #                   )


if __name__ == '__main__':
    main()
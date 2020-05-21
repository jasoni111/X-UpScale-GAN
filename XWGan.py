import tensorflow as tf
import numpy as np

# import matplotlib.pyplot as plt
# import PIL
from data import getAnimeCleanData, getCelebaData
from loss import (
    w_d_loss,
    w_g_loss,
    gradient_penalty,
    # generator_loss,
    # discriminator_loss,
    cycle_loss,
    identity_loss,
    mse_loss,
    gradient_penalty_star,
)
from discriminator import StarDiscriminator, W_Discriminator
from functools import partial
from c_dann import C_dann
from encoder import *

# , UpScaleDiscriminator
from datetime import datetime

batch_size = 32


def run_tensorflow():
    """
    [summary] This is needed for tensorflow to free up my gpu ram...
    """

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

    mixed_precision = tf.keras.mixed_precision.experimental

    policy = mixed_precision.Policy("mixed_float16")
    mixed_precision.set_policy(policy)

    AnimeCleanData = getAnimeCleanData(BATCH_SIZE=batch_size)
    CelebaData = getCelebaData(BATCH_SIZE=batch_size)

    logdir = "./logs/XWGan/" + datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(logdir)

    checkpoint_path = "./checkpoints/XWGan"

    encode_anime = encoder_seperate_layers()
    encode_human = encoder_seperate_layers()
    encode_share = encoder_shared_layers()

    decode_share = decoder_shared_layers()
    decode_human = decoder_seperate_layers()
    decode_anime = decoder_seperate_layers()
    c_dann = C_dann()
    D = W_Discriminator()

    optims = [
        mixed_precision.LossScaleOptimizer(
            tf.keras.optimizers.Adam(1e-4, beta_1=0.9, beta_2=0.99), loss_scale="dynamic"
        )
        for _ in range(8)
    ]
    # gan_optim = mixed_precision.LossScaleOptimizer(
    #     tf.keras.optimizers.Adam(1e-4, beta_1=0.5), loss_scale="dynamic"
    # )
    # dis_optim = mixed_precision.LossScaleOptimizer(
    #     tf.keras.optimizers.Adam(1e-4, beta_1=0.5), loss_scale="dynamic"
    # )

    ckpt = tf.train.Checkpoint(
        encode_anime=encode_anime,
        encode_human=encode_human,
        encode_share=encode_share,
        decode_share=decode_share,
        decode_human=decode_human,
        decode_anime=decode_anime,
        c_dann=c_dann,
    )

    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    # if a checkpoint exists, restore the latest checkpoint.
    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print("Latest checkpoint restored!!")

    @tf.function
    def trainstep(real_human, real_anime, big_anime):
        with tf.GradientTape(persistent=True) as tape:
            latent_anime = encode_share(encode_anime(real_anime))
            latent_human = encode_share(encode_human(real_human))

            recon_anime = decode_anime(decode_share(latent_anime))
            recon_human = decode_human(decode_share(latent_human))

            fake_anime = decode_anime(decode_share(latent_human))
            latent_human_cycled = encode_share(encode_anime(fake_anime))

            fake_human = decode_anime(decode_share(latent_anime))
            latent_anime_cycled = encode_share(encode_anime(fake_human))

            def kl_loss(mean, log_var):
                loss = 1 + log_var - tf.math.square(mean) + tf.math.exp(log_var)
                loss = tf.reduce_sum(loss, axis=-1) * -0.5
                return loss

            disc_fake = D(fake_anime)
            disc_real = D(real_anime)

            c_dann_anime = c_dann(latent_anime)
            c_dann_human = c_dann(latent_human)

            loss_anime_encode = identity_loss(real_anime, recon_anime) * 3
            loss_human_encode = identity_loss(real_human, recon_human) * 3


            loss_domain_adversarial = tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.zeros_like(c_dann_anime), logits=c_dann_anime
                )
            ) + tf.reduce_mean(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.ones_like(c_dann_human), logits=c_dann_human
                )
            )
            loss_domain_adversarial = tf.math.minimum(loss_domain_adversarial, 100)
            loss_domain_adversarial = loss_domain_adversarial * 0.2
            tf.print(loss_domain_adversarial)

            loss_semantic_consistency = (
                identity_loss(latent_anime, latent_anime_cycled) * 3
                + identity_loss(latent_human, latent_human_cycled) * 3
            )

            loss_gan = w_g_loss(disc_fake)

            anime_encode_total_loss = (
                loss_anime_encode
                + loss_domain_adversarial
                + loss_semantic_consistency
                + loss_gan
            )
            human_encode_total_loss = (
                loss_human_encode
                + loss_domain_adversarial
                + loss_semantic_consistency
            )
            share_encode_total_loss = (
                loss_anime_encode
                + loss_domain_adversarial
                + loss_semantic_consistency
                + loss_gan
                + loss_human_encode
            )

            share_decode_total_loss = loss_anime_encode + loss_human_encode + loss_gan
            anime_decode_total_loss = loss_anime_encode + loss_gan
            human_decode_total_loss = loss_human_encode


            # loss_disc = (
            #     mse_loss(tf.ones_like(disc_fake), disc_fake)
            #     + mse_loss(tf.zeros_like(disc_real), disc_real)
            # ) * 10
            loss_disc = w_d_loss(disc_real,disc_fake)
            loss_disc += gradient_penalty(partial(D,training=True),real_anime,fake_anime)

            losses = [anime_encode_total_loss,human_encode_total_loss,share_encode_total_loss,loss_domain_adversarial,share_decode_total_loss,
            anime_decode_total_loss,human_decode_total_loss,loss_disc]

            scaled_losses = [optim.get_scaled_loss(loss) for optim,loss in zip(optims,losses)]

        list_variables = [
            encode_anime.trainable_variables,
            encode_human.trainable_variables,
            encode_share.trainable_variables,
            c_dann.trainable_variables,
            decode_share.trainable_variables,
            decode_anime.trainable_variables,
            decode_human.trainable_variables,
            D.trainable_variables
        ]
        gan_grad = [tape.gradient(scaled_loss, train_variable) for scaled_loss, train_variable in zip(scaled_losses,list_variables)  ]
        gan_grad = [optim.get_unscaled_gradients(x) for optim,x in zip(optims,gan_grad)]
        for optim, grad, trainable in zip(optims,gan_grad, list_variables):
            optim.apply_gradients(zip(grad, trainable))
        # dis_grad = dis_optim.get_unscaled_gradients(
        #     tape.gradient(scaled_loss_disc, D.trainable_variables)
        # )
        # dis_optim.apply_gradients(zip(dis_grad, D.trainable_variables))

        return (
            real_human,
            real_anime,
            recon_anime,
            recon_human,
            fake_anime,
            fake_human,
            loss_anime_encode,
            loss_human_encode,
            loss_domain_adversarial,
            loss_semantic_consistency,
            loss_gan,
            loss_disc,
            tf.reduce_mean(disc_fake),
            tf.reduce_mean(disc_real)
        )

    def process_data_for_display(input_image):
        return input_image * 0.5 + 0.5

    print_string = [
        "real_human",
        "real_anime",
        "recon_anime",
        "recon_human",
        "fake_anime",
        "fake_human",
        "loss_anime_encode",
        "loss_human_encode",
        "loss_domain_adversarial",
        "loss_semantic_consistency",
        "loss_gan",
        "loss_disc",
        "disc_fake",
        "disc_real"
    ]

    counter = 0
    i = -1
    while True:
        i = i + 1
        counter = counter + 1
        AnimeBatchImage, BigAnimeBatchImage = next(iter(AnimeCleanData))
        CelebaBatchImage = next(iter(CelebaData))
        print(counter)

        if not (i % 5):
            result = trainstep(CelebaBatchImage, AnimeBatchImage, BigAnimeBatchImage)

            with file_writer.as_default():
                for j in range(len(result)):

                    if j < 6:
                        tf.summary.image(
                            print_string[j],
                            process_data_for_display(result[j]),
                            step=counter,
                        )
                    else:
                        print(print_string[j], result[j])
                        tf.summary.scalar(
                            print_string[j], result[j], step=counter,
                        )

            ckpt_manager.save()
        else:
            trainstep(CelebaBatchImage, AnimeBatchImage, BigAnimeBatchImage)


# testfun()
run_tensorflow()

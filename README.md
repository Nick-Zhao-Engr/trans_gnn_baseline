## structure learning baseline for trans_gnn project

lds config:
cora hyperparameter:
configs = LDSConfig.grid(pat=20, seed=seed, io_steps=20,
                        io_lr=(2.e-2, 1.e-4, 0.05), keep_prob=0.5,
                        oo_lr=(1., 1., 1.e-3), hidden=32)

citeseer:
configs = LDSConfig.grid(pat=20, seed=seed, io_steps=20,
                                 io_lr=(2.e-2, 1.e-4, 0.05), keep_prob=0.5,
                                 oo_lr=(.1, 1., 1.e-3),hidden=16)

amherst:
LDSConfig.grid(pat=20, seed=seed, io_steps=20,
                                 io_lr=(2.e-2, 1.e-4, 0.05), keep_prob=0.5,
                                 oo_lr=(.1, 1., 1.e-3),hidden=16)

jh:

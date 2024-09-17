""" Scheduler Factory
Hacked together by / Copyright 2020 Ross Wightman
"""
from timm.scheduler.cosine_lr import CosineLRScheduler
from timm.scheduler.tanh_lr import TanhLRScheduler
from timm.scheduler.step_lr import StepLRScheduler
from timm.scheduler.plateau_lr import PlateauLRScheduler
from timm.scheduler.multistep_lr import MultiStepLRScheduler
from timm.scheduler.poly_lr import PolyLRScheduler


def create_scheduler(args, optimizer):
    num_epochs = args.epochs

    t_in_epochs = not args.sched_in_steps
    if t_in_epochs:
        warmup_t = args.warmup_epochs
        t_initial = num_epochs
        decay_t = args.decay_epochs
        cooldown_t = args.cooldown_epochs
        decay_milestones = args.decay_milestones
    else:
        warmup_t = args.warmup_epochs * args.num_steps_epoch
        t_initial = num_epochs * args.num_steps_epoch
        decay_t = args.decay_epochs * args.num_steps_epoch
        cooldown_t = args.cooldown_epochs * args.num_steps_epoch
        decay_milestones = [d * args.num_steps_epoch for d in args.decay_milestones]
    if t_initial == 0:
        t_initial = 1
        
    # warmup args
    warmup_args = dict(
        warmup_lr_init=args.warmup_lr,
        warmup_t=warmup_t,
        warmup_prefix=False,
    )
    

    if getattr(args, 'lr_noise', None) is not None:
        lr_noise = getattr(args, 'lr_noise')
        if isinstance(lr_noise, (list, tuple)):
            noise_range = [n * num_epochs for n in lr_noise]
            if len(noise_range) == 1:
                noise_range = noise_range[0]
        else:
            noise_range = lr_noise * num_epochs
    else:
        noise_range = None

    noise_args = dict(
        noise_range_t=noise_range,
        noise_pct=args.lr_noise_pct,
        noise_std=args.lr_noise_std,
        noise_seed=args.seed,
    )

    # setup cycle args for supporting schedulers
    cycle_args = dict(
        cycle_mul=args.lr_cycle_mul,
        cycle_decay=args.decay_rate,
        cycle_limit=1,
    )

    k_decay = 1.0


    lr_scheduler = None
    if args.sched == 'cosine':
        lr_scheduler = CosineLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=args.min_lr,
            t_in_epochs=t_in_epochs,
            **cycle_args,
            **warmup_args,
            **noise_args,
            k_decay=k_decay,
        )
    elif args.sched == 'tanh':
        lr_scheduler = TanhLRScheduler(
            optimizer,
            t_initial=t_initial,
            lr_min=args.min_lr,
            t_in_epochs=t_in_epochs,
            **cycle_args,
            **warmup_args,
            **noise_args,
        )
    elif args.sched == 'step':
        lr_scheduler = StepLRScheduler(
            optimizer,
            decay_t=decay_t,
            decay_rate=args.decay_rate,
            t_in_epochs=t_in_epochs,
            **warmup_args,
            **noise_args,
        )
    elif args.sched == 'multistep':
        lr_scheduler = MultiStepLRScheduler(
            optimizer,
            decay_t=decay_milestones,
            decay_rate=args.decay_rate,
            t_in_epochs=t_in_epochs,
            **warmup_args,
            **noise_args,
        )
    elif args.sched == 'plateau':
        assert t_in_epochs, 'Plateau LR only supports step per epoch.'
        warmup_args.pop('warmup_prefix', False)
        lr_scheduler = PlateauLRScheduler(
            optimizer,
            decay_rate=args.decay_rate,
            patience_t=args.patience_epochs,
            cooldown_t=0,
            **warmup_args,
            lr_min=args.min_lr,
            mode=args.plateau_mode,
            **noise_args,
        )
    elif args.sched == 'poly':
        lr_scheduler = PolyLRScheduler(
            optimizer,
            power=args.decay_rate,  # overloading 'decay_rate' as polynomial power
            t_initial=t_initial,
            lr_min=args.min_lr,
            t_in_epochs=t_in_epochs,
            k_decay=k_decay,
            **cycle_args,
            **warmup_args,
            **noise_args,
        )

    if hasattr(lr_scheduler, 'get_cycle_length'):
        # for cycle based schedulers (cosine, tanh, poly) recalculate total epochs w/ cycles & cooldown
        t_with_cycles_and_cooldown = lr_scheduler.get_cycle_length() + cooldown_t
        if t_in_epochs:
            num_epochs = t_with_cycles_and_cooldown
        else:
            num_epochs = t_with_cycles_and_cooldown // args.num_steps_epoch

    return lr_scheduler, num_epochs

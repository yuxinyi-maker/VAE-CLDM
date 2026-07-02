
import numpy as np
import torch
import math

class LambdaWarmUpCosineScheduler:
    """
    Single-cycle cosine annealing learning rate scheduler 
    """
 
    def __init__(self, warm_up_steps, lr_min, lr_max, lr_start, max_decay_steps, verbosity_interval=0):
        """
        Initialize single-cycle cosine annealing scheduler - text
        
        Args:
            warm_up_steps: textsteps
            lr_min: minimum learning rate
            lr_max: maximum learning rate  
            lr_start: start learning rate
            max_decay_steps: textsteps
            verbosity_interval: logging interval
        """
        self.lr_warm_up_steps = warm_up_steps
        self.lr_start = lr_start
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.
        self.verbosity_interval = verbosity_interval
        
        # Key fix:textparameterstext
        self._validate_and_fix_parameters()
        
        print("🔧 textSingle-cycle cosine annealing learning rate scheduler (text)")
        print(f"  📈 textsteps: {warm_up_steps}")
        print(f"  🎯 learning rate range: [{self.lr_min:.2e}, {self.lr_max:.2e}]")
        print(f"  🚀 start learning rate: {self.lr_start:.2e}")
        print(f"  ⏱️  textsteps: {max_decay_steps}")

    def _validate_and_fix_parameters(self):
        """textparameters - Key fixtext"""
        # textminimum learning ratetext
        if self.lr_max <= self.lr_min:
            print(f"⚠️ textlearning rate range: lr_max({self.lr_max:.2e}) <= lr_min({self.lr_min:.2e})")
            # textlr_max > lr_min
            self.lr_max = max(self.lr_min * 10.0, 1e-4)
            print(f"  ✅ text: lr_max = {self.lr_max:.2e}")
        
        # textstart learning ratetext
        if self.lr_start <= 0:
            self.lr_start = self.lr_min
            print(f"⚠️ textstart learning rate: {self.lr_start:.2e}")
            
        # textstepstext
        if self.lr_warm_up_steps <= 0:
            self.lr_warm_up_steps = max(1000, self.lr_max_decay_steps // 20)
            print(f"⚠️ textsteps: {self.lr_warm_up_steps}")
            
        # textstepstextsteps
        if self.lr_max_decay_steps <= self.lr_warm_up_steps:
            self.lr_max_decay_steps = self.lr_warm_up_steps * 10
            print(f"⚠️ textsteps: {self.lr_max_decay_steps}")
            
        # textlearning ratetext
        self.lr_min = max(self.lr_min, 1e-8)
        self.lr_max = max(self.lr_max, 1e-8)
        self.lr_start = max(self.lr_start, 1e-8)

    def schedule(self, n, **kwargs):
        """Compute learning rate multiplier at step n"""
        if self.verbosity_interval > 0 and n % self.verbosity_interval == 0:
            print(f"📊 Learning rate schedule - steps: {n}, current multiplier: {self.last_lr:.6f}")

        # Warmup phase
        if n < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
            self.last_lr = lr
            return lr
        
        # Cosine annealing phase
        else:
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)
            
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + math.cos(t * math.pi))
            lr = max(lr, self.lr_min)
            self.last_lr = lr
            
            return lr

    def __call__(self, n, **kwargs):
        """Make instance callable"""
        return self.schedule(n, **kwargs)

    def get_current_lr_info(self):
        """textlearning ratetext"""
        return {
            'last_lr': self.last_lr,
            'warm_up_steps': self.lr_warm_up_steps,
            'lr_range': [self.lr_min, self.lr_max],
            'total_steps': self.lr_max_decay_steps
        }


class LambdaWarmUpCosineScheduler2:
    """
    textcycletextLearning rate scheduletext - text
    """

    def __init__(self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):
        """
        textcycletext 
        
        Args:
            warm_up_steps: textcycletextstepstext
            f_min: textcycleminimum learning ratetext
            f_max: textcyclemaximum learning ratetext  
            f_start: textcyclestart learning ratetext
            cycle_lengths: textcycletext
            verbosity_interval: logging interval
        """
        # text
        assert len(warm_up_steps) == len(f_min) == len(f_max) == len(f_start) == len(cycle_lengths)
        
        if len(warm_up_steps) == 0:
            raise ValueError("parameterstext")

        self.lr_warm_up_steps = warm_up_steps
        self.f_start = f_start
        self.f_min = f_min
        self.f_max = f_max
        self.cycle_lengths = cycle_lengths
        
        # textsteps
        self.cum_cycles = np.cumsum([0] + list(self.cycle_lengths))
        self.last_f = 0.
        self.verbosity_interval = verbosity_interval
        
        # Key fix:textparameterstext
        self._validate_and_fix_parameters()
        
        print("🔧 textcycletextLearning rate scheduletext (text)")
        print(f"  📊 cycle count: {len(cycle_lengths)}")
        print(f"  🎯 learning rate range: {[[self.f_min[i], self.f_max[i]] for i in range(len(self.f_min))]}")

    def _validate_and_fix_parameters(self):
        
        for i in range(len(self.cycle_lengths)):
            # textminimum learning ratetext
            if self.f_max[i] <= self.f_min[i]:
                print(f"⚠️ cycle{i}: textlearning rate range f_max({self.f_max[i]}) <= f_min({self.f_min[i]})")
                self.f_max[i] = max(self.f_min[i] * 1.1, self.f_min[i] + 0.1)
                print(f"  ✅ text: f_max = {self.f_max[i]}")
            
            # textstart learning rate
            if self.f_start[i] <= 0:
                self.f_start[i] = self.f_min[i]
                print(f"⚠️ cycle{i}: textstart learning ratetext {self.f_start[i]}")
            
            # textsteps
            if self.lr_warm_up_steps[i] <= 0:
                self.lr_warm_up_steps[i] = max(100, self.cycle_lengths[i] // 20)
                print(f"⚠️ cycle{i}: textstepstext {self.lr_warm_up_steps[i]}")
                
            # textcycletext
            if self.cycle_lengths[i] <= self.lr_warm_up_steps[i]:
                self.cycle_lengths[i] = self.lr_warm_up_steps[i] * 2
                print(f"⚠️ cycle{i}: textcycletext {self.cycle_lengths[i]}")
                
            # textlearning ratetext
            self.f_min[i] = max(self.f_min[i], 1e-8)
            self.f_max[i] = max(self.f_max[i], 1e-8)
            self.f_start[i] = max(self.f_start[i], 1e-8)

    def find_in_interval(self, n):
        """textstepstextcycle"""
        if n >= self.cum_cycles[-1]:
            return len(self.cycle_lengths) - 1
            
        for interval in range(len(self.cum_cycles) - 1):
            if self.cum_cycles[interval] <= n < self.cum_cycles[interval + 1]:
                return interval
                
        return len(self.cycle_lengths) - 1

    def schedule(self, n, **kwargs):
        """Compute learning rate multiplier at step n"""
        cycle = self.find_in_interval(n)
        n_in_cycle = n - self.cum_cycles[cycle]

        if self.verbosity_interval > 0 and n_in_cycle % self.verbosity_interval == 0:
            print(f"📊 Learning rate schedule - textsteps: {n}, cycle: {cycle}, current multiplier: {self.last_f:.6f}")

        # Warmup phase
        if n_in_cycle < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n_in_cycle + self.f_start[cycle]
            self.last_f = f
            return f
            
        # Cosine annealing phase
        else:
            t = (n_in_cycle - self.lr_warm_up_steps[cycle]) / (self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle])
            t = min(t, 1.0)
            
            f = self.f_min[cycle] + 0.5 * (self.f_max[cycle] - self.f_min[cycle]) * (1 + math.cos(t * math.pi))
            f = max(f, self.f_min[cycle])
            self.last_f = f
            
            return f

    def __call__(self, n, **kwargs):
        """Make instance callable"""
        return self.schedule(n, **kwargs)

    def get_current_cycle_info(self, n):
        """textstepstextcycletext"""
        cycle = self.find_in_interval(n)
        n_in_cycle = n - self.cum_cycles[cycle]
        
        return {
            'total_step': n,
            'current_cycle': cycle,
            'step_in_cycle': n_in_cycle,
            'cycle_total_steps': self.cycle_lengths[cycle],
            'warm_up_steps': self.lr_warm_up_steps[cycle],
            'lr_range': [self.f_min[cycle], self.f_max[cycle]]
        }


class LambdaLinearScheduler(LambdaWarmUpCosineScheduler2):
    
    
    def __init__(self, warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval=0):
        """textcycletext"""
        super().__init__(warm_up_steps, f_min, f_max, f_start, cycle_lengths, verbosity_interval)
        print("🔧 textcycletextLearning rate scheduletext (text)")

    def schedule(self, n, **kwargs):
        """Linear decay schedule strategy"""
        cycle = self.find_in_interval(n)
        n_in_cycle = n - self.cum_cycles[cycle]

        if self.verbosity_interval > 0 and n_in_cycle % self.verbosity_interval == 0:
            print(f"📊 Linear schedule - textsteps: {n}, cycle: {cycle}, current multiplier: {self.last_f:.6f}")

        # Warmup phase
        if n_in_cycle < self.lr_warm_up_steps[cycle]:
            f = (self.f_max[cycle] - self.f_start[cycle]) / self.lr_warm_up_steps[cycle] * n_in_cycle + self.f_start[cycle]
            self.last_f = f
            return f
            
        # textphase
        else:
            decay_progress = (n_in_cycle - self.lr_warm_up_steps[cycle]) / \
                           (self.cycle_lengths[cycle] - self.lr_warm_up_steps[cycle])
            decay_progress = min(decay_progress, 1.0)
            
            f = self.f_max[cycle] - (self.f_max[cycle] - self.f_min[cycle]) * decay_progress
            f = max(f, self.f_min[cycle])
            self.last_f = f
            
            return f


class Rock3DConditionalScheduler:
    
    
    def __init__(self, base_lr=5.0e-5, total_steps=100000, warmup_ratio=0.1, 
                 min_lr_ratio=0.01, strategy='cosine', verbosity_interval=1000):
       
        self.base_lr = base_lr
        self.total_steps = total_steps
        self.warmup_steps = int(total_steps * warmup_ratio)
        self.min_lr = base_lr * min_lr_ratio
        self.strategy = strategy
        self.verbosity_interval = verbosity_interval
        self.current_step = 0
        self.current_lr = 0.0
        
        # parameterstext
        self._validate_and_fix_parameters()
        
        print("🎯 text3DtextLearning rate scheduletext")
        print(f"  📊 base learning rate: {self.base_lr:.2e}")
        print(f"  ⏱️  textsteps: {total_steps}")
        print(f"  🔥 textsteps: {self.warmup_steps} ({warmup_ratio*100:.1f}%)")
        print(f"  🎯 minimum learning rate: {self.min_lr:.2e}")
        print(f"  📈 schedule strategy: {strategy}")

    def _validate_and_fix_parameters(self):
        """textparameters - Key fixtext"""
        # textbase learning ratetext
        if self.base_lr <= 0:
            self.base_lr = 5.0e-5
            print(f"⚠️ textbase learning rate: {self.base_lr:.2e}")
            
        # textminimum learning ratetextbase learning rate
        if self.min_lr <= 0:
            self.min_lr = self.base_lr * 0.01
            print(f"⚠️ textminimum learning rate: {self.min_lr:.2e}")
            
        if self.min_lr >= self.base_lr:
            self.min_lr = self.base_lr * 0.1
            print(f"⚠️ textminimum learning ratetext: {self.min_lr:.2e}")
            
        # textstepstext
        if self.warmup_steps <= 0:
            self.warmup_steps = max(1000, self.total_steps // 20)
            print(f"⚠️ textsteps: {self.warmup_steps}")
            
        # textstepstext
        if self.total_steps <= self.warmup_steps:
            self.total_steps = self.warmup_steps * 10
            print(f"⚠️ textsteps: {self.total_steps}")
            
        # text
        if self.strategy not in ['cosine', 'linear']:
            self.strategy = 'cosine'
            print(f"⚠️ textschedule strategy: {self.strategy}")

    def get_lr(self, step):
        """textstepstextlearning rate"""
        self.current_step = step
        
        # Warmup phase
        if step < self.warmup_steps:
            lr = self.base_lr * (step / self.warmup_steps)
        else:
            # main training phase
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(progress, 1.0)
            
            if self.strategy == 'cosine':
                lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (1 + math.cos(progress * math.pi))
            else:
                lr = self.base_lr - (self.base_lr - self.min_lr) * progress
        
        lr = max(lr, self.min_lr)
        self.current_lr = lr
        
        if self.verbosity_interval > 0 and step % self.verbosity_interval == 0:
            phase = "warmup" if step < self.warmup_steps else "main training"
            print(f"📊 3Dtext - steps: {step}, phase: {phase}, learning rate: {lr:.2e}")
            
        return lr

    def __call__(self, step):
        """Make instance callable"""
        return self.get_lr(step)

    def get_schedule_info(self):
        """text"""
        return {
            'current_step': self.current_step,
            'current_lr': self.current_lr,
            'base_lr': self.base_lr,
            'min_lr': self.min_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps,
            'strategy': self.strategy,
            'progress': f"{(self.current_step/self.total_steps)*100:.1f}%"
        }


# text
def create_safe_scheduler(config, total_steps=None):
   
    print("🔧 textLearning rate scheduletext...")
    
    # text - text3Dtext
    default_config = {
        'target': 'ldm.lr_scheduler_new.LambdaWarmUpCosineScheduler',
        'params': {
            'warm_up_steps': 5000,
            'lr_min': 1e-6,
            'lr_max': 5e-4,
            'lr_start': 1e-6,
            'max_decay_steps': 100000,
            'verbosity_interval': 1000
        }
    }
    
    # textvalue
    scheduler_config = config.get('scheduler_config', default_config)
    params = scheduler_config.get('params', default_config['params'])
    
    target = scheduler_config.get('target', default_config['target'])
    
    print(f"  🎯 scheduler type: {target}")
    
    try:
        # text
        if target == 'ldm.lr_scheduler_new.LambdaWarmUpCosineScheduler':
            scheduler = LambdaWarmUpCosineScheduler(**params)
        elif target == 'ldm.lr_scheduler_new.LambdaWarmUpCosineScheduler2':
            scheduler = LambdaWarmUpCosineScheduler2(**params)
        elif target == 'ldm.lr_scheduler_new.LambdaLinearScheduler':
            scheduler = LambdaLinearScheduler(**params)
        elif target == 'ldm.lr_scheduler_new.Rock3DConditionalScheduler':
            if total_steps:
                params['total_steps'] = total_steps
            scheduler = Rock3DConditionalScheduler(**params)
        else:
            print(f"⚠️ textscheduler type: {target},text")
            scheduler = LambdaWarmUpCosineScheduler(**default_config['params'])
            
        print("✅ textsuccess")
        return scheduler
        
    except Exception as e:
        print(f"❌ textfailed: {e},text")
        # text - text
        return LambdaWarmUpCosineScheduler(
            warm_up_steps=5000,
            lr_min=1e-6,
            lr_max=5e-4,
            lr_start=1e-6,
            max_decay_steps=100000
        )


def validate_scheduler_consistency(model_config, scheduler):
    
    base_lr = model_config.get('base_learning_rate', 5.0e-5)
    
    # textbase learning rate(text)
    scheduler_base_lr = getattr(scheduler, 'base_lr', getattr(scheduler, 'lr_max', None))
    
    if scheduler_base_lr is not None and abs(scheduler_base_lr - base_lr) > 1e-8:
        print(f"⚠️ text: textbase learning rate({scheduler_base_lr:.2e})text({base_lr:.2e})text")
        return False
    else:
        print("✅ learning ratetext")
        return True


# text
def test_schedulers():
    """text"""
    print("🧪 textLearning rate scheduletext...")
    
    try:
        # textcycletext
        scheduler1 = LambdaWarmUpCosineScheduler(
            warm_up_steps=100,
            lr_min=1e-6,
            lr_max=1e-4,
            lr_start=1e-6,
            max_decay_steps=1000
        )
        lr1 = scheduler1(50)
        print(f"✅ textcycletext: lr={lr1:.2e}")
        
        # textcycletext
        scheduler2 = LambdaWarmUpCosineScheduler2(
            warm_up_steps=[100, 50],
            f_min=[1e-6, 1e-7],
            f_max=[1e-4, 1e-5],
            f_start=[1e-6, 1e-7],
            cycle_lengths=[1000, 500]
        )
        lr2 = scheduler2(1200)
        print(f"✅ textcycletext: lr={lr2:.2e}")
        
        # text3Dtext
        scheduler3 = Rock3DConditionalScheduler(
            base_lr=5e-5,
            total_steps=1000,
            warmup_ratio=0.1,
            min_lr_ratio=0.01,
            strategy='cosine'
        )
        lr3 = scheduler3(500)
        print(f"✅ 3Dtext: lr={lr3:.2e}")
        
        print("✅ All scheduler tests passed!")
        
    except Exception as e:
        print(f"❌ textfailed: {e}")
        return False
        
    return True


if __name__ == "__main__":
    # text
    test_schedulers()
"""
Conditional training monitor callback
Specifically used to monitor condition control effectiveness
"""

import torch
import pytorch_lightning as pl


class ConditionMonitor(pl.Callback):
    """Condition control monitor callback"""
    
    def __init__(self, monitor_porosity=True, monitor_rock_type=True):
        super().__init__()
        self.monitor_porosity = monitor_porosity
        self.monitor_rock_type = monitor_rock_type
        self.best_porosity_mae = float('inf')
        
    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        """Monitor condition consistency at validation batch end"""
        if outputs is None or not isinstance(outputs, dict):
            return
            
        # porosity condition
        if self.monitor_porosity and 'val_porosity_mae' in outputs:
            porosity_mae = outputs['val_porosity_mae']
            if torch.isfinite(porosity_mae):
                pl_module.log("val_porosity_mae_batch", porosity_mae, on_step=False, on_epoch=True)
                
        # Monitor generated porosity range
        if 'val_generated_porosity' in outputs and 'val_target_porosity' in outputs:
            gen_porosity = outputs['val_generated_porosity']
            target_porosity = outputs['val_target_porosity']
            if torch.isfinite(gen_porosity) and torch.isfinite(target_porosity):
                porosity_bias = torch.abs(gen_porosity - target_porosity)
                pl_module.log("val_porosity_bias", porosity_bias, on_step=False, on_epoch=True)
    
    def on_validation_epoch_end(self, trainer, pl_module):
        """Report condition consistency at validation epoch end"""
        metrics = trainer.callback_metrics
        
        print(f"\n🔍 Condition control monitor report (Epoch {trainer.current_epoch}):")
        
        # porosity condition
        if self.monitor_porosity:
            if 'val_porosity_mae' in metrics:
                mae = metrics['val_porosity_mae'].item()
                status = "✅ Excellent" if mae < 0.05 else "⚠️ Good" if mae < 0.1 else "🚨 Needs improvement"
                print(f"  🕳️  Porosity MAE: {mae:.4f} ({status})")
                
            if 'val_porosity_bias' in metrics:
                bias = metrics['val_porosity_bias'].item()
                print(f"  📊 Porosity bias: {bias:.4f}")
                
            if 'val_generated_porosity' in metrics and 'val_target_porosity' in metrics:
                gen = metrics['val_generated_porosity'].item()
                target = metrics['val_target_porosity'].item()
                print(f"  🎯 Generated porosity: {gen:.3f}, target porosity: {target:.3f}")
        
        # Save best condition consistency model
        if 'val_porosity_mae' in metrics:
            current_mae = metrics['val_porosity_mae'].item()
            if current_mae < self.best_porosity_mae:
                self.best_porosity_mae = current_mae
                print(f"🎉 New best condition consistency: {current_mae:.4f}")
                
    def on_train_epoch_start(self, trainer, pl_module):
        """Check condition network status at training epoch start"""
        # Check whether condition projection network is trainable
        if hasattr(pl_module, 'condition_projection'):
            trainable_params = sum(p.numel() for p in pl_module.condition_projection.parameters() if p.requires_grad)
            frozen_params = sum(p.numel() for p in pl_module.condition_projection.parameters() if not p.requires_grad)
            
            if trainer.current_epoch % 10 == 0:
                print(f"🔧 Condition projection network: {trainable_params:,} Parameters, {frozen_params:,} Parameters")
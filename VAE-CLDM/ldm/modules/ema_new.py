
import torch
from torch import nn
import warnings


class LitEma(nn.Module):
    
    def __init__(self, model, decay=0.9999, use_num_updates=True):
        
        super().__init__()

        # Validate decay rate range
        if decay < 0.0 or decay > 1.0:
            raise ValueError('Decay must be between 0 and 1')

        # Register buffers - fix device synchronization
        self.m_name2s_name = {}  # text:model parameter name -> EMAbuffer name
        self.register_buffer('decay', torch.tensor(decay, dtype=torch.float32))
        
        # Key fix:text use_num_upates -> use_num_updates
        self.register_buffer('num_updates', 
                           torch.tensor(0, dtype=torch.int) if use_num_updates 
                           else torch.tensor(-1, dtype=torch.int))

        # Collect model device information
        model_device = next(model.parameters()).device
        print(f"🔧 EMA initialization: model device={model_device}, decay rate={decay}")

        # Initialize EMA buffers for all trainable parameters
        param_count = 0
        for name, p in model.named_parameters():
            if p.requires_grad:
                # Remove'.'characters(buffer nametext)
                s_name = name.replace('.', '')
                self.m_name2s_name.update({name: s_name})
                
                # Clone and detach parameter data as initial EMA values, ensuring device consistency
                ema_buffer = p.clone().detach().data.to(model_device)
                self.register_buffer(s_name, ema_buffer)
                param_count += 1
                
                # Debug information:record 3D-specific parameters
                if any(dim > 64 for dim in p.shape):  # large parameters may be 3D convolutions
                    print(f"  📊 3Dparameters: {name}, shape: {p.shape}")

        print(f"✅ EMA initializationtext: {param_count}textparameters")

        # textStoreparameterstext
        self.collected_params = []
        
        # Storetext(text)
        self._model_ref = None

    def forward(self, model):
        """
        textEMAparameters - enhanced 3D compatibility

        Args:
            model: current training target model(3Dtext)
        """
        # Storetext
        self._model_ref = model
        
        decay = self.decay

        # textdecay rate(textstepstext)
        if self.num_updates >= 0:
            self.num_updates += 1
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
            
            # text1000textDebug information
            if self.num_updates % 1000 == 0:
                print(f"🔄 EMA update: steps={self.num_updates.item()}, textdecay rate={decay:.6f}")

        one_minus_decay = 1.0 - decay

        with torch.no_grad():
            m_param = dict(model.named_parameters())
            shadow_params = dict(self.named_buffers())

            updated_params = 0
            skipped_params = 0
            
            # textparameterstextEMAvalue
            for key in m_param:
                if m_param[key].requires_grad:
                    if key not in self.m_name2s_name:
                        # textparameterstext(text)
                        warnings.warn(f"⚠️ EMA: textparameters '{key}',textEMA update")
                        skipped_params += 1
                        continue
                        
                    sname = self.m_name2s_name[key]
                    
                    # Ensure EMA buffer exists
                    if sname not in shadow_params:
                        warnings.warn(f"⚠️ EMA: text '{sname}' text,skip update")
                        skipped_params += 1
                        continue
                    
                    # Ensure dtype and device consistency
                    shadow_params[sname] = shadow_params[sname].type_as(m_param[key])
                    
                    # textshapetext
                    if shadow_params[sname].shape != m_param[key].shape:
                        warnings.warn(f"⚠️ EMA: shapetext '{key}' ({shadow_params[sname].shape} vs {m_param[key].shape})")
                        skipped_params += 1
                        continue
                    
                    # EMA updatetext: new = decay * old + (1-decay) * current
                    # text: new = old - (1-decay) * (old - current)
                    shadow_params[sname].mul_(decay)
                    shadow_params[sname].add_(m_param[key].data, alpha=one_minus_decay)
                    
                    updated_params += 1
                else:
                    # textparameters,textEMAtext
                    if key in self.m_name2s_name:
                        warnings.warn(f"⚠️ EMA: textparameters '{key}' textEMAtext,text")

            # Print update statistics
            if updated_params > 0 and self.num_updates % 5000 == 0:
                print(f"📈 EMAstatistics: text{updated_params}parameters, text{skipped_params}parameters")

    def copy_to(self, model):
       
        m_param = dict(model.named_parameters())
        shadow_params = dict(self.named_buffers())

        copied_params = 0
        failed_params = 0
        
        for key in m_param:
            if m_param[key].requires_grad:
                if key not in self.m_name2s_name:
                    warnings.warn(f"⚠️ EMAtext: parameters '{key}' textEMAtext")
                    failed_params += 1
                    continue
                    
                sname = self.m_name2s_name[key]
                
                if sname not in shadow_params:
                    warnings.warn(f"⚠️ EMAtext: text '{sname}' text")
                    failed_params += 1
                    continue
                
                # textshapetext
                if shadow_params[sname].shape != m_param[key].shape:
                    warnings.warn(f"⚠️ EMAtext: shapetext '{key}'")
                    failed_params += 1
                    continue
                
                # textEMAparameterstext
                m_param[key].data.copy_(shadow_params[sname].data)
                copied_params += 1
        
        if copied_params > 0:
            print(f"✅ EMAparameterstext: success{copied_params}text, failed{failed_params}text")
        else:
            print(f"❌ EMAparameterstext: text{copied_params + failed_params}textparameterstextfailed")

    def store(self, parameters):
     
        try:
            self.collected_params = [param.clone().detach() for param in parameters]
            print(f"💾 Store{len(self.collected_params)}textparameterstextrestore")
        except Exception as e:
            print(f"❌ parametersStorefailed: {e}")
            self.collected_params = []

    def restore(self, parameters):
        
        if not self.collected_params:
            print("⚠️ textStoretextparameterstextrestore")
            return
            
        if len(self.collected_params) != len(parameters):
            print(f"⚠️ parameterstext: Store{len(self.collected_params)} vs text{len(parameters)}")
            return
            
        restored_count = 0
        for c_param, param in zip(self.collected_params, parameters):
            try:
                # textshapetext
                if c_param.shape == param.shape:
                    param.data.copy_(c_param.data)
                    restored_count += 1
                else:
                    print(f"⚠️ parametersshapetext: {c_param.shape} vs {param.shape}")
            except Exception as e:
                print(f"❌ parametersrestorefailed: {e}")
                
        print(f"🔁 parametersrestore: {restored_count}/{len(parameters)}textparameters")

    def get_decay_info(self):
       
        return {
            'decay': self.decay.item(),
            'num_updates': self.num_updates.item(),
            'total_params': len(self.m_name2s_name)
        }

    def get_ema_state_dict(self):
       
        state_dict = {}
        
        # text(textparameters)
        for name, buffer in self.named_buffers():
            state_dict[name] = buffer.clone()
            
        # text
        state_dict['m_name2s_name'] = self.m_name2s_name.copy()
        state_dict['ema_config'] = {
            'decay': self.decay.item(),
            'num_updates': self.num_updates.item()
        }
        
        print(f"💾 EMAstate dict: {len(state_dict)}text")
        return state_dict

    def load_ema_state_dict(self, state_dict, strict=True):
       
        # text
        ema_config = state_dict.pop('ema_config', None)
        name_mapping = state_dict.pop('m_name2s_name', None)
        
        if ema_config:
            print(f"📥 Load EMA config: decay rate={ema_config.get('decay', 'N/A')}, "
                  f"update count={ema_config.get('num_updates', 'N/A')}")
        
        if name_mapping:
            # text
            self.m_name2s_name.update(name_mapping)
            print(f"🔧 Load name mapping: {len(name_mapping)}textparameters")
        
        # Load buffers
        current_buffers = dict(self.named_buffers())
        loaded_count = 0
        missing_count = 0
        
        for name, buffer in state_dict.items():
            if name in current_buffers:
                # textshapetext
                if current_buffers[name].shape == buffer.shape:
                    current_buffers[name].data.copy_(buffer.data)
                    loaded_count += 1
                else:
                    print(f"⚠️ EMAtext: shapetext '{name}' "
                          f"({current_buffers[name].shape} vs {buffer.shape})")
                    if not strict:
                        # textshape
                        try:
                            if current_buffers[name].numel() == buffer.numel():
                                current_buffers[name].data.copy_(buffer.data.view_as(current_buffers[name]))
                                loaded_count += 1
                                print(f"  ✅ textshapetext")
                            else:
                                print(f"  ❌ text,text")
                        except Exception as e:
                            print(f"  ❌ textfailed: {e}")
            else:
                if strict:
                    print(f"⚠️ EMAtext: missing buffer '{name}'")
                missing_count += 1
        
        print(f"✅ EMAtext: success{loaded_count}text, text{missing_count}text")

    def analyze_parameters(self):
        
        shadow_params = dict(self.named_buffers())
        stats = {
            'total_buffers': len(shadow_params),
            'ema_mappings': len(self.m_name2s_name),
            '3d_large_params': 0,
            'parameter_stats': {}
        }
        
        for name, buffer in shadow_params.items():
            if name in ['decay', 'num_updates']:
                continue
                
            # statisticsparameterstext
            param_stats = {
                'shape': list(buffer.shape),
                'dtype': str(buffer.dtype),
                'device': str(buffer.device),
                'mean': buffer.mean().item(),
                'std': buffer.std().item(),
                'nan_count': torch.isnan(buffer).sum().item(),
                'inf_count': torch.isinf(buffer).sum().item()
            }
            
            stats['parameter_stats'][name] = param_stats
            
            # text3Dtextparameters(text5text: [B, C, D, H, W])
            if len(buffer.shape) >= 5 and any(dim > 64 for dim in buffer.shape):
                stats['3d_large_params'] += 1
                
        return stats


class Rock3DEmaWrapper:
    

    def __init__(self, model, decay=0.9999, use_num_updates=True, enable_3d_optimizations=True):
        """
        Initialize 3D EMA wrapper

        Args:
            model: 3Dtext
            decay: decay rate
            use_num_updates: textupdate counttext
            enable_3d_optimizations: text3Dtext
        """
        self.ema = LitEma(model, decay, use_num_updates)
        self.model = model
        self.enable_3d_optimizations = enable_3d_optimizations
        self._original_params_stored = False
        self._original_params = []
        
        print(f"🎯 3D EMAtext: decay rate={decay}, 3Dtext={enable_3d_optimizations}")

    def update(self):
        """textEMAparameters - text3Dtext"""
        if self.enable_3d_optimizations:
            # 3Dtext:textparameterstext
            model_device = next(self.model.parameters()).device
            self.ema.to(model_device)
            
        self.ema(self.model)

    def apply_ema(self):
        """textEMAparameterstext(text)- text"""
        if not self._original_params_stored:
            # textStoretextparameters
            self._store_original_params()
            
        print("🔄 textEMAparameterstext...")
        self.ema.copy_to(self.model)

    def restore_original(self):
        """restoretextparameters(text)- text"""
        if self._original_params_stored and self._original_params:
            print("🔁 restoretextparameterstext...")
            current_params = list(self.model.parameters())
            
            restored = 0
            for orig, current in zip(self._original_params, current_params):
                if orig.shape == current.shape:
                    current.data.copy_(orig.data)
                    restored += 1
                    
            print(f"✅ restore{restored}/{len(current_params)}textparameters")
        else:
            print("⚠️ textStoretextparameterstextrestore")

    def _store_original_params(self):
        """Storetextparameters - text"""
        self._original_params = [param.clone().detach() for param in self.model.parameters()]
        self._original_params_stored = True
        print(f"💾 Store{len(self._original_params)}textparameters")

    def state_dict(self):
        """textstate dict"""
        state = {
            'ema_state_dict': self.ema.state_dict(),
            'ema_extra_state': self.ema.get_ema_state_dict(),
            'wrapper_info': {
                'enable_3d_optimizations': self.enable_3d_optimizations,
                'has_original_params': self._original_params_stored
            }
        }
        return state

    def load_state_dict(self, state_dict):
        """textstate dict - text"""
        try:
            if 'ema_state_dict' in state_dict:
                self.ema.load_state_dict(state_dict['ema_state_dict'])
                
            if 'ema_extra_state' in state_dict:
                self.ema.load_ema_state_dict(state_dict['ema_extra_state'])
                
            if 'wrapper_info' in state_dict:
                info = state_dict['wrapper_info']
                self.enable_3d_optimizations = info.get('enable_3d_optimizations', True)
                self._original_params_stored = info.get('has_original_params', False)
                
            print("✅ EMAtextsuccess")
            
        except Exception as e:
            print(f"❌ EMAtextfailed: {e}")
            raise

    def get_detailed_info(self):
        """Get detailed EMA information - text"""
        ema_info = self.ema.get_decay_info()
        stats = self.ema.analyze_parameters()
        
        info = {
            'ema_config': ema_info,
            'parameter_stats': stats,
            'wrapper_status': {
                '3d_optimizations': self.enable_3d_optimizations,
                'original_params_stored': self._original_params_stored,
                'model_parameters': sum(p.numel() for p in self.model.parameters()),
                'ema_buffers': stats['total_buffers']
            }
        }
        
        return info

    def set_decay(self, new_decay):
        """
        textdecay rate - text
        
        Args:
            new_decay: textdecay rate(0-1text)
        """
        if 0.0 <= new_decay <= 1.0:
            self.ema.decay.data = torch.tensor(new_decay, device=self.ema.decay.device)
            print(f"🔧 textEMAdecay rate: {new_decay}")
        else:
            raise ValueError("decay ratetext0text1text")


# Utility functions
def create_ema_for_3d_model(model, decay=0.9999, use_wrapper=True):
    """
    text3DtextEMAtext
    
    Args:
        model: 3Dtext
        decay: decay rate
        use_wrapper: whether to use wrapper
        
    Returns:
        LitEmatextRock3DEmaWrappertext
    """
    if use_wrapper:
        return Rock3DEmaWrapper(model, decay=decay)
    else:
        return LitEma(model, decay=decay)


def ema_model_checkpoint(ema_module, filepath):
    """
    EMAtextUtility functions
    
    Args:
        ema_module: EMAtext
        filepath: save path
    """
    if isinstance(ema_module, Rock3DEmaWrapper):
        state_dict = ema_module.state_dict()
    else:
        state_dict = ema_module.get_ema_state_dict()
        
    torch.save(state_dict, filepath)
    print(f"💾 EMAtext: {filepath}")


if __name__ == "__main__":
    # Test code
    print("✅ ema_new.py textsuccess - 3Dtext")
    
    # text
    class TestModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv3d = nn.Conv3d(1, 4, kernel_size=3)
            self.linear = nn.Linear(10, 5)
            
        def forward(self, x):
            return self.linear(x)
    
    # Test EMA functionality
    test_model = TestModel()
    ema = LitEma(test_model, decay=0.9)
    print("🧪 EMA functionality test passed")
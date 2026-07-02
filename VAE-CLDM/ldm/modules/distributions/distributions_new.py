

import torch
import numpy as np


class AbstractDistribution:
    """
    Abstract base class for probability distributions
    Provides a unified interface for different probability distributions
    """

    def sample(self):
        """Sample from distribution"""
        raise NotImplementedError()

    def mode(self):
        """Get distribution mode(most likely value)"""
        raise NotImplementedError()

    def log_prob(self, x):
        """Compute log probability"""
        raise NotImplementedError()


class DiracDistribution(AbstractDistribution):
    """
    Dirac distribution(deterministic distribution),all probability mass is concentrated at a single point
    used for deterministic encoding or decoding
    """

    def __init__(self, value):
        """
        English textDirac distribution

        Args:
            value: unique value of the distribution
        """
        self.value = value

    def sample(self):
        """Sampling(always returns the same value)"""
        return self.value

    def mode(self):
        """Mode(English textSamplingEnglish text)"""
        return self.value

    def log_prob(self, x):
        """Log probability(infinite at value and negative infinite elsewhere)"""
        # English text,actualEnglish text
        return torch.zeros_like(x) if torch.allclose(x, self.value) else -torch.ones_like(x) * float('inf')




class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False, spatial_dims=3):
        """
        Initialize diagonal Gaussian distribution - greatly reduce debug information
        """
        self.parameters = parameters
        # Split mean and log variance
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)

        self.logvar = torch.clamp(self.logvar, -6.0, 2.0)
        
        self.deterministic = deterministic
        self.spatial_dims = spatial_dims

        if self.deterministic:
            # Deterministic mode:variance is 0
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)
        else:
            # Stochastic mode:compute standard deviation and variance
            self.std = torch.exp(0.5 * self.logvar)
            self.var = torch.exp(self.logvar)

    
        self._setup_dimensions()


        import os
        debug_env = os.getenv('VAE_DEBUG', '0')
        if debug_env == '1' and not hasattr(self, '_debug_printed_global'):
            # Set global flag to ensure output only once
            DiagonalGaussianDistribution._debug_printed_global = True
            print(f"🔧 Distribution initialization debug (global):")
            print(f"  English textshape: {self.parameters.shape}")
            print(f"  English textshape: {self.mean.shape}") 
            print(f"  Spatial dimensions: {self.spatial_dims}")
            print(f"  KL sum dimensions: {self.kl_dims}")

    def _setup_dimensions(self):
        """
        Set correct dimension list for summation - silent version
        """
        
        self.kl_dims = list(range(1, 1 + self.spatial_dims + 1))  # 3D: [1,2,3,4]

        # Debug info - shown only on first initialization
        if hasattr(self, 'mean') and not hasattr(self, '_debug_printed'):
            print(f"🔧 Distribution initialization debug:")
            print(f"  English textshape: {self.parameters.shape}")
            print(f"  English textshape: {self.mean.shape}") 
            print(f"  Spatial dimensions: {self.spatial_dims}")
            print(f"  KL sum dimensions: {self.kl_dims}")
            self._debug_printed = True



  
    def kl(self, other=None):
        """Compute KL divergence - fix over-constraint issue"""
        if self.deterministic:
            return torch.tensor(0., device=self.parameters.device)
    
        try:
            if other is None:
                # KL divergence with standard normal distribution N(0,1)
                kl_elements = 0.5 * (
                    torch.pow(self.mean, 2) + self.var - 1.0 - self.logvar
                )

                kl_loss = torch.sum(kl_elements, dim=self.kl_dims)
                kl_loss = torch.mean(kl_loss)
                
              
                kl_loss = torch.clamp(kl_loss, min=1e-8, max=100.0)
                
                return kl_loss
                
            else:
                kl_elements = 0.5 * (
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar
                )
                
                kl_loss = torch.sum(kl_elements, dim=self.kl_dims)
                kl_loss = torch.mean(kl_loss)
                kl_loss = torch.clamp(kl_loss, min=1e-8, max=50.0)
                
                return kl_loss
                
        except Exception as e:
            print(f"🚨 KL loss computation exception: {e}")
            return torch.tensor(1e-4, device=self.parameters.device)

    def sample(self):
       
        if self.deterministic: # key fix:Deterministic modeEnglish text
            sampled = self.mean
        else:
          
            sampled = self.mean + self.std * torch.randn_like(self.mean)
        
        
        sampled = torch.tanh(sampled) * 1.5
        return sampled

    def mode(self):
        """
        English textMode - English text
        
        key fix:English textModeEnglish text,English textSamplingEnglish text
        """
        
        mode = self.mean
       
        mode = torch.tanh(mode) * 1.5
        return mode

    def nll(self, sample, dims=None):
        
        if self.deterministic:
            return torch.tensor(0., device=self.parameters.device)

        if dims is None:
            dims = self.kl_dims

        logtwopi = np.log(2.0 * np.pi)
        nll_elements = 0.5 * (
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var
        )
        
        nll_loss = torch.sum(nll_elements, dim=dims)
        nll_loss = torch.mean(nll_loss)
        
        return nll_loss

    def log_prob(self, x):
        """
        Compute log probability - English text
        """
        if self.deterministic:
            return torch.zeros_like(x)

        log_prob_elements = -0.5 * (
            np.log(2.0 * np.pi) + self.logvar + torch.pow(x - self.mean, 2) / self.var
        )
        
        # Sum over spatial and channel dimensions, average over batch dimension
        log_prob = torch.sum(log_prob_elements, dim=self.kl_dims)
        log_prob = torch.mean(log_prob)
        
        return log_prob

    def analyze_kl_components(self):
        """
        Analyze KL loss components - new debug method
        
        Used to monitor KL loss components and help diagnose issues
        Return values of each KL loss component

        Returns:
            dict: dictionary containing KL loss components
        """
        if self.deterministic:
            return {
                'mean_sq': 0.0,
                'var_component': 0.0, 
                'neg_logvar': 0.0,
                'constant': -1.0,
                'estimated_kl': 0.0
            }
        
        try:
            # Compute each KL loss component
            mean_sq = torch.mean(torch.pow(self.mean, 2))
            var_component = torch.mean(self.var)
            neg_logvar = torch.mean(-self.logvar)
            constant = -1.0
            
            # Estimate KL loss(English textactualEnglish text)
            estimated_kl = 0.5 * (mean_sq + var_component + neg_logvar + constant)
            
            return {
                'mean_sq': mean_sq.item(),
                'var_component': var_component.item(),
                'neg_logvar': neg_logvar.item(), 
                'constant': constant,
                'estimated_kl': estimated_kl.item()
            }
        except Exception as e:
            print(f"KL component analysis exception: {e}")
            return {
                'mean_sq': 0.0,
                'var_component': 0.0,
                'neg_logvar': 0.0,
                'constant': -1.0,
                'estimated_kl': 0.0
            }


def normal_kl(mean1, logvar1, mean2, logvar2, spatial_dims=3):
   
    tensor = None
    for obj in (mean1, logvar1, mean2, logvar2):
        if isinstance(obj, torch.Tensor):
            tensor = obj
            break

    assert tensor is not None, "at least one argument must be a Tensor"

    # Ensure all parameters are tensors
    logvar1, logvar2 = [
        x if isinstance(x, torch.Tensor) else torch.tensor(x).to(tensor)
        for x in (logvar1, logvar2)
    ]

    # KL divergence formula
    kl_elements = 0.5 * (
        -1.0
        + logvar2
        - logvar1
        + torch.exp(logvar1 - logvar2)
        + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
    )
    
    
    kl_dims = list(range(1, 1 + spatial_dims + 1))
    kl_loss = torch.sum(kl_elements, dim=kl_dims)
    

    kl_loss = torch.mean(kl_loss)
    
    # Numerical stability constraint
    kl_loss = torch.clamp(kl_loss, min=1e-8, max=10.0)
    
    return kl_loss



class Rock3DDistributionUtils:
    
    @staticmethod
    def create_prior_distribution(shape, device='cuda'):
       
        # CreateEnglish text
        batch_size, channels, depth, height, width = shape
        mean = torch.zeros(shape, device=device)
        logvar = torch.zeros(shape, device=device)

        # Concatenate parameters
        parameters = torch.cat([mean, logvar], dim=1)
        return DiagonalGaussianDistribution(parameters, deterministic=False)

    @staticmethod
    def create_safe_distribution(parameters, deterministic=False):
       
        try:
          
            parameters = torch.tanh(parameters) * 2.0
            
            dist = DiagonalGaussianDistribution(
                parameters, 
                deterministic=deterministic,
                spatial_dims=3
            )
            return dist
        except Exception as e:
            print(f"🚨 English textCreateEnglish text: {e}")
          
            device = parameters.device
            shape = parameters.shape
            safe_params = torch.zeros(shape, device=device)
            return DiagonalGaussianDistribution(safe_params, deterministic=True)

    @staticmethod
    def monitor_distribution_stats(distribution, step=None):
       
        if not hasattr(distribution, 'mean'):
            return {}
        
        try:
            stats = {
                'mean_range': [distribution.mean.min().item(), distribution.mean.max().item()],
                'logvar_range': [distribution.logvar.min().item(), distribution.logvar.max().item()],
                'var_range': [distribution.var.min().item(), distribution.var.max().item()],
                'std_range': [distribution.std.min().item(), distribution.std.max().item()],
            }
            
            if step is not None:
                print(f"📊 stepsEnglish text {step} distribution statistics:")
                print(f"  English textrange: [{stats['mean_range'][0]:.3f}, {stats['mean_range'][1]:.3f}]")
                print(f"  English textrange: [{stats['logvar_range'][0]:.3f}, {stats['logvar_range'][1]:.3f}]")
                print(f"  English textrange: [{stats['var_range'][0]:.6f}, {stats['var_range'][1]:.6f}]")
                print(f"  English textrange: [{stats['std_range'][0]:.6f}, {stats['std_range'][1]:.6f}]")
            
            return stats
        except Exception as e:
            print(f"distribution statisticsEnglish text: {e}")
            return {}

    @staticmethod
    def bernoulli_log_prob(x, probs, eps=1e-8):
      
        probs = torch.clamp(probs, eps, 1 - eps)
        return x * torch.log(probs) + (1 - x) * torch.log(1 - probs)

    @staticmethod
    def binary_kl_divergence(q_probs, p_probs, eps=1e-8):
  
        q_probs = torch.clamp(q_probs, eps, 1 - eps)
        p_probs = torch.clamp(p_probs, eps, 1 - eps)

        return q_probs * torch.log(q_probs / p_probs) + (1 - q_probs) * torch.log((1 - q_probs) / (1 - p_probs))

    @staticmethod
    def validate_distribution(distribution, threshold=10.0):
      
        try:
            if not hasattr(distribution, 'mean'):
                return False
            
            # English textrange
            mean_max = distribution.mean.max().item()
            mean_min = distribution.mean.min().item()
            
            # English textrange
            var_max = distribution.var.max().item()
            
            # If any parameter exceeds the threshold, consider distribution unhealthy
            if (abs(mean_max) > threshold or 
                abs(mean_min) > threshold or 
                var_max > threshold):
                print(f"⚠️ Distribution validation failed: English text[{mean_min:.3f}, {mean_max:.3f}], max variance{var_max:.3f}")
                return False
                
            return True
            
        except Exception as e:
            print(f"Distribution validation exception: {e}")
            return False
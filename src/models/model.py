import jax.numpy as jnp
import flax.linen as nn

class VGG16(nn.Module):
    num_classes: int
    dropout_rate: float = 0.2
    output: str='softmax'
    dtype: str='float32'
        
    @nn.compact
    def __call__(self, x, training=False):
        
        if self.output not in ['softmax', 'log_softmax', 'sigmoid', 'linear', 'log_sigmoid']:
            raise ValueError('Wrong argument. Possible choices for output are "softmax", "sigmoid", "log_sigmoid", "linear".')
        
        x = self._Conv_Block(x, features=64, num_layers=2, dtype=self.dtype)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = self._Conv_Block(x, features=128, num_layers=2, dtype=self.dtype)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = self._Conv_Block(x, features=256, num_layers=3, dtype=self.dtype)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = self._Conv_Block(x, features=512, num_layers=3, dtype=self.dtype)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        x = self._Conv_Block(x, features=512, num_layers=3, dtype=self.dtype)
        x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        
        # Fully conected
        x = self._GlobalAvgPool2D(x)
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.BatchNorm()(x, use_running_average=not training)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)
        x = nn.Dense(features=self.num_classes)(x)   
        if self.output == 'softmax':
          x = nn.softmax(x)
        if self.output == 'log_softmax':
          x = nn.log_softmax(x)
        if self.output == 'sigmoid':
          x = nn.sigmoid(x)
        if self.output == 'log_sigmoid':
          x = nn.log_sigmoid(x)
        return x
    
    def _GlobalAvgPool2D(self, inputs):
        x = jnp.mean(inputs, axis=(2, 3))
        return x
    
    def _Conv_Block(self, x, features, num_layers, dtype):
        for l in range(num_layers):
            x = nn.Conv(features=features, kernel_size=(3, 3), padding='same', dtype=dtype)(x)
            x = nn.relu(x)
        return x
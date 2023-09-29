from utils import *

class Node():
    def __init__(self,
                 name,
                 activation='sigmoid',
                 input_weight=None,
                 bias=0,
                 iter=-1,
                 iter_g=-1,
                #  is_output=False,
                 loss_function=None) -> None:
        self.name=name
        self.activation=activation
        self.bias=bias
        self.iter=iter
        self.iter_g=iter_g
        self.input_weight=input_weight
        self.upstream_nodes_dict={} # {Node: weight}
        self.downstream_nodes_dict={} # {Node: weight}
        self.value=None
        self.z=None
        self.grad=None
        self.grad_z=None
        self.grad_weights=None
        self.grad_bias=None
        self.grad_input_weight=None
        # self.is_output=is_output
        self.loss_function=loss_function
        self.loss=None
        self.local_loss=None
        return
    
    def __eq__(self, another):
        return hasattr(another, 'name') and self.name == another.name
    
    def __hash__(self):
        return hash(self.name)
    


    def add_upstream_node(self, node, weight):
        if not node in self.upstream_nodes_dict:
            self.upstream_nodes_dict[node] = weight
            node.add_downstream_node(self, weight)
        return

    def add_downstream_node(self, node, weight):
        if not node in self.downstream_nodes_dict:
            self.downstream_nodes_dict[node] = weight
            node.add_upstream_node(self, weight)
        return
    
    def get_z(self, iter=None, input=None, verbose=False):
        if iter is None or iter <= self.iter:
            # if verbose:
            #     print(f"{self.name}: z already calculated.")
            return self.z
        
        if verbose:
            print(f"{self.name}: caluclating z ...")

        z = self.bias
        if self.input_weight is not None:
            assert len(input.shape) == 2
            assert input.shape[1] == self.input_weight.shape[0]
            # z += sum(input * self.input_weight)
            z += np.dot(input, self.input_weight)

        for node, weight in self.upstream_nodes_dict.items():
            if verbose:
                print(f"{self.name}: requesting value from node {node.name} ...")
            z += node.get_value(iter, input=input, verbose=verbose) * weight
        self.z = z
        assert len(z.shape) == 1
        return self.z
    
    def get_value(self, iter=None, input=None, verbose=False):
        if iter is None or iter <= self.iter:
            # if verbose:
            #     print(f"{self.name}: value already calculated.")
            return self.value
        
        if verbose:
            print(f"{self.name}: caluclating value ...")

        if self.activation == 'sigmoid':
            self.value = sigmoid(self.get_z(iter, input=input, verbose=verbose))
        elif self.activation == 'relu':
            self.value = relu(self.get_z(iter, input=input, verbose=verbose))
        else:
            raise Exception('activation function not recognized')
        self.iter = iter
        assert len(self.value.shape) == 1
        return self.value
    
    def get_loss(self, input, output, iter=None):
        self.get_value(iter=iter, input=input)
        if self.loss_function == 'squared_error':
            self.loss =  (self.value - output)**2
        elif self.loss_function == 'binary_logistic_regression':
            self.loss =  -1 * output * np.log(self.value) - (1.0 - output) * np.log(1.0 - self.value)
        else:
            raise Exception(f'loss function {self.loss_function} not recognized.')
        assert len(self.loss.shape) == 1
        assert self.loss.shape == output.shape
        return self.loss
        
    def get_grad(self, iter_g=None, output=None, verbose=False):
        if iter_g is None or iter_g <= self.iter_g:
            # if verbose:
            #     print(f"{self.name}: grad already calculated.")
            return self.grad
        
        if verbose:
            print(f"{self.name}: caluclating grad ...")

        # if self.is_output:
        #     if output is None:
        #         raise Exception('output is required for the last node.')
        #     self.grad = 2 * (self.get_value(verbose=verbose) - output)
        if self.loss_function is not None:
            if output is None:
                raise Exception('output is required for the last node.')
            elif self.loss_function == 'squared_error':
                self.grad = 2 * (self.get_value(verbose=verbose) - output)
            elif self.loss_function == 'binary_logistic_regression':
                self.grad = -1 * output / self.get_value(verbose=verbose) + (1.0 - output) / (1.0 - self.get_value(verbose=verbose))
            else:
                raise Exception (f'{node.name}: loss function not recognized.')
        else:
            g_v = 0.0
            for node, weight in self.downstream_nodes_dict.items():
                if verbose:
                    print(f"{self.name}: requesting grad-z from node {node.name} ...")
                g_v += node.get_grad_z(iter_g, output=output, verbose=verbose) * weight
            self.grad = g_v
            
        self.local_loss = relu(self.grad) * self.value + relu(-1*self.grad) * (1.0 - self.value)
        return self.grad

    def get_grad_z(self, iter_g=None, output=None, verbose=False):
        if iter_g is None or iter_g <= self.iter_g:
            # if verbose:
            #     print(f"{self.name}: grad-z already calculated.")
            return self.grad_z
        
        if verbose:
            print(f"{self.name}: calculating grad-z ...")

        if self.activation == 'sigmoid':
            self.grad_z = self.get_grad(iter_g, output=output, verbose=verbose) * self.value * (1.0 - self.value)
        elif self.activation == 'relu':
            # TODO: allow for batch update
            self.grad_z = ((self.value > 0)*1.0) * self.get_grad(iter_g, output=output, verbose=verbose)
            # if self.value > 0:
            #     self.grad_z = self.get_grad(iter_g, output=output, verbose=verbose)
            # else:
            #     self.grad_z = 0.0
        else:
            raise Exception('activation function not recognized')
        
        self.iter_g = iter_g
        return self.grad_z
        
    def get_grad_weights(self, iter_g=None, verbose=False):
        if iter_g is None:
            # if verbose:
            #     print(f"{self.name}: grad-w already calculated.")
            return self.grad_weights

        if verbose:
            print(f"{self.name}: calculating grad-w ...")

        grad_z = self.get_grad_z(iter_g, verbose=verbose)

        self.grad_weights = {}
        for node in self.upstream_nodes_dict:
            if verbose:
                print(f"{self.name}: requesting value from node {node.name} ...")
            self.grad_weights[node] = grad_z * node.get_value(verbose=verbose)

        return self.grad_weights
    
    def get_grad_bias(self, iter_g=None, verbose=False):
        if iter_g is None:
            # if verbose:
            #     print(f"{self.name}: grad-b already calculated.")
            return self.grad_bias
        
        if verbose:
            print(f"{self.name}: calculating grad-b ...")
        self.grad_bias = self.get_grad_z(iter_g, verbose=verbose)
        return self.grad_bias
    
    def get_grad_input_weight(self, input, iter_g=None, verbose=False):
        if iter_g is None or input is None:
            # if verbose:
            #     print(f"{self.name}: grad-input-w already calculated.")
            return self.grad_input_weight
        
        if verbose:
            print(f"{self.name}: calculating grad-input-w ...")
        if self.input_weight is not None:
            self.grad_input_weight = self.get_grad_z(iter_g, verbose=verbose).reshape(-1,1) * input
        return self.grad_input_weight

    def get_all_grads(self, iter_g, input, output, verbose=False):
        return {
            'grad_z': self.get_grad_z(iter_g=iter_g, output=output, verbose=verbose),
            'grad_v': self.get_grad(iter_g=iter_g, output=output, verbose=verbose),
            'grad_w': self.get_grad_weights(iter_g=iter_g, verbose=verbose),
            'grad_b': self.get_grad_bias(iter_g=iter_g, verbose=verbose),
            'grad_input_w': self.get_grad_input_weight(input=input, iter_g=iter_g, verbose=verbose)
            }
    
    def take_step(self, step_size=0.1, l2_reg=0.0):
        self.bias = self.bias - step_size * (self.grad_bias.mean(axis=0) + l2_reg * self.bias)
        for node, grad_w in self.grad_weights.items():
            self.upstream_nodes_dict[node] = self.upstream_nodes_dict[node] - step_size * (grad_w.mean(axis=0) + l2_reg * self.upstream_nodes_dict[node])
            node.downstream_nodes_dict[self] = self.upstream_nodes_dict[node]
            
        if self.input_weight is not None:
            self.input_weight = self.input_weight - step_size * (self.grad_input_weight.mean(axis=0) + l2_reg * self.input_weight)
        return

function [J grad] = nnCostFunction(nn_params, input_layer_size, hidden_layer_size, num_labels, X, y, lambda)

Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), num_labels, (hidden_layer_size + 1));

m = size(X, 1);
% Converting labels to a m*10 matrix.
Y = zeros(m,size(Theta2));
for iter = 1:m
  Y(iter,y(iter)) = 1;
end

h2 = sigmoid([ones(m, 1) X] * Theta1');
h3 = sigmoid([ones(m, 1) h2] * Theta2');
% calculating cost
J = sum(sum(-Y.*log(h3) - (1-Y).*log(1-h3)))/m + (sum(sum(Theta1(:,2:end).*Theta1(:,2:end))) + sum(sum(Theta2(:,2:end).*Theta2(:,2:end))) )*lambda/(2*m);

% calculating grad
tri1 = zeros(size(Theta1));
tri2 = zeros(size(Theta2));
for i = 1:m
  a1 = X(i,:)'; a1 = [1;a1];
  z2 = Theta1*a1;
  a2 = [1;sigmoid(z2)];
  a3 = sigmoid(Theta2*a2);
  delta3 = a3 - Y(i,:)';
  delta21 = (Theta2'*delta3);
  delta2 = delta21(2:size(delta21,1),:).*sigmoidGradient(z2);
  tri1 = tri1 + delta2*a1';
  tri2 = tri2 + delta3*a2';
end

Theta2_grad = tri2/m + lambda*[zeros(size(Theta2,1),1) Theta2(:,2:end)]/m;
Theta1_grad = tri1/m + lambda*[zeros(size(Theta1,1),1) Theta1(:,2:end)]/m;

grad = [Theta1_grad(:) ; Theta2_grad(:)];

end

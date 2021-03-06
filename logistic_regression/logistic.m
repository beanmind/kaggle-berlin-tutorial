%{
A = textread("data/train.csv", "%d", "delimiter", ",");
B = textread("data/train.csv", "%s", "delimiter", ",");
inds = isnan(A);
B(!inds) = num2cell(A(!inds));

%}

data = csvread ('../data/titanic_train_modified.csv');
test = csvread ('../data/titanic_test_modified.csv');

fprintf('loading the data \n');
% Training set
X = data(2:800, 3:end); 
y = data(2:800, 1);

%Cross validation CV set
CV = data(801:end, 3:end); 
CVy = data(801:end, 1);

% Test set
T = test(2:end, 2:end);

%  Setup the data matrix appropriately, and add ones for the intercept term
[m, n] = size(X);
[o,p] = size(y);
[q,r] = size (T);
[s,t] = size (CV);

fprintf (' m= %f \n n=%f \n', m, n);
fprintf (' o= %f \n p=%f \n\n', o, p);
fprintf (' q= %f \n r=%f \n\n', q, r);

% Add intercept term to x and T_test
X = [ones(m, 1) X];
T = [ones(q, 1) T];
CV = [ones(s,1) CV];

% Initialize fitting parameters
initial_theta = zeros(n + 1, 1);

% Compute and display initial cost and gradient
[cost, grad] = costFunction(initial_theta, X, y);

fprintf('Cost at initial theta (zeros): %f\n', cost);
fprintf('Gradient at initial theta (zeros): \n');
fprintf(' %f \n', grad);

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%  Set options for fminunc
options = optimset('GradObj', 'on', 'MaxIter', 400);

%  Run fminunc to obtain the optimal theta
%  This function will return theta and the cost 
[theta, cost] = ...
	fminunc(@(t)(costFunction(t, X, y)), initial_theta, options);

% Print theta to screen
fprintf('Cost at theta found by fminunc: %f\n', cost);
fprintf('theta: \n');
fprintf(' %f \n', theta);

%{
% Plot Boundary
plotDecisionBoundary(theta, X, y);

% Put some labels 
hold on;
% Labels and Legend
xlabel('Exam 1 score')
ylabel('Exam 2 score')

% Specified in plot order
legend('Admitted', 'Not admitted')
hold off;

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ============== Part 4: Predict and Accuracies ==============
%  After learning the parameters, you'll like to use it to predict the outcomes
%  on unseen data. In this part, you will use the logistic regression model
%  to predict the probability that a student with score 45 on exam 1 and 
%  score 85 on exam 2 will be admitted.
%
%  Furthermore, you will compute the training and test set accuracies of 
%  our model.
%
%  Your task is to complete the code in predict.m

%  Predict probability for a student with score 45 on exam 1 
%  and score 85 on exam 2 
%}

prob = sigmoid( CV(3,:)* theta);
fprintf(['For a student with scores 45 and 85, we predict an admission ' ...
         'probability of %f\n\n'], prob);

% Compute accuracy on our training set
p = predict(theta, X);

fprintf('Train Accuracy: %f\n', mean(double(p == y)) * 100);

% Compute accuracy on our CV set
p = predict(theta, CV);
fprintf('Train Accuracy: %f\n', mean(double(p == CVy)) * 100);


fprintf('\nProgram paused. Press enter to continue.\n');
pause;

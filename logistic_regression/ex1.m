%{
A = textread("data/train.csv", "%d", "delimiter", ",");
B = textread("data/train.csv", "%s", "delimiter", ",");
inds = isnan(A);
B(!inds) = num2cell(A(!inds));

%}

x = csvread ('../data/titanic_train_modified.csv');





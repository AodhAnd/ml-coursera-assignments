%% ============== Part 6: Entering ratings for a new user ===============
%  Before we will train the collaborative filtering model, we will first
%  add ratings that correspond to a new user that we just observed. This
%  part of the code will also allow you to put in your own ratings for the
%  movies in our dataset!
%
movieList = loadMovieList();

%  Initialize my ratings
my_ratings = zeros(1682, 1);

% Check the file movie_idx.txt for id of each movie in our dataset
% For example, Toy Story (1995) has ID 1, so to rate it "4", you can set
my_ratings(1) = 4;

% We have selected a few movies we liked / did not like and the ratings we
% gave are as follows:
my_ratings(2) = 3.5;
my_ratings(12)= 5;
my_ratings(50) = 4.5;
my_ratings(56)= 4.5;
my_ratings(66)= 3;
my_ratings(71) = 3.5;
my_ratings(95) = 4.5;
my_ratings(168) = 4.5;
my_ratings(172)= 5;
my_ratings(178)= 5;
my_ratings(181)= 5;
my_ratings(186)= 5;
my_ratings(193)= 3;
my_ratings(196)= 3;
my_ratings(204)= 3.5;
my_ratings(214)= 5;
my_ratings(250)= 4;
my_ratings(271)= 2.5;
my_ratings(272)= 3.5;
my_ratings(313)= 2.5;
my_ratings(475)= 4;
my_ratings(902)= 4;
my_ratings(1472)= 2.5;


fprintf('\n\nNew user ratings:\n');
for i = 1:length(my_ratings)
    if my_ratings(i) > 0 
        fprintf('Rated %d for %s\n', my_ratings(i), ...
                 movieList{i});
    end
end

fprintf('\nProgram paused. Press enter to continue.\n');
pause;


%% ================== Part 7: Learning Movie Ratings ====================
%  Now, you will train the collaborative filtering model on a movie rating 
%  dataset of 1682 movies and 943 users
%

fprintf('\nTraining collaborative filtering...\n');

%  Load data
load('ex8_movies.mat');

%  Y is a 1682x943 matrix, containing ratings (1-5) of 1682 movies by 
%  943 users
%
%  R is a 1682x943 matrix, where R(i,j) = 1 if and only if user j gave a
%  rating to movie i

%  Add our own ratings to the data matrix
Y = [my_ratings Y];
R = [(my_ratings ~= 0) R];

%  Normalize Ratings
[Ynorm, Ymean] = normalizeRatings(Y, R);

%  Useful Values
num_users = size(Y, 2);
num_movies = size(Y, 1);
num_features = 10;

% Set Initial Parameters (Theta, X)
X = randn(num_movies, num_features);
Theta = randn(num_users, num_features);

initial_parameters = [X(:); Theta(:)];

% Set options for fmincg
options = optimset('GradObj', 'on', 'MaxIter', 100);

% Set Regularization
lambda = 10;
theta = fmincg (@(t)(cofiCostFunc(t, Ynorm, R, num_users, num_movies, ...
                                num_features, lambda)), ...
                initial_parameters, options);

% Unfold the returned theta back into U and W
X = reshape(theta(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(theta(num_movies*num_features+1:end), ...
                num_users, num_features);

fprintf('Recommender system learning completed.\n');

fprintf('\nProgram paused. Press enter to continue.\n');
pause;

%% ================== Part 8: Recommendation for you ====================
%  After training the model, you can now make recommendations by computing
%  the predictions matrix.
%

p = X * Theta';
my_predictions = p(:,1) + Ymean;

movieList = loadMovieList();

[r, ix] = sort(my_predictions, 'descend');
fprintf('\nTop recommendations for you:\n');
for i=1:15
    j = ix(i);
    fprintf('Predicting rating %.1f for movie %s\n', my_predictions(j), ...
            movieList{j});
end

function [X, length, all_means] = loadMovie(tplay)
    % LOADMOVIE Converts each frame of a video to grayscale
    % Filters grayscale to focus on white components to track bucket
    % movement. Finds coordinate locations of focused white components to
    % track movement. Takes the mean of x and y coordinates and puts them
    % into one large matrix of the averaged coordinates, which the function
    % returns. Also returns the length of the final matrix for use in
    % further analysis. 
    
    % implay(tplay); % optional method to play the video
    all_means = [];
    numFrames = size(tplay,4);
    for j = 1:numFrames
        X = tplay(:,:,:,j);
        X = double(rgb2gray(X)); % convert each video frame to grayscale
        white = X > 240;
        location = find(white);
        [y, x]= ind2sub(size(white), location);
        y_mean = mean(y);
        x_mean = mean(x);
        all_means = [all_means; y_mean, x_mean];
        all_means = rmmissing(all_means); % remove NaN values from data
       % imshow(white)
       % drawnow
    end
    [rows, ~] = size(all_means);
    length = rows;
    X = X;
end


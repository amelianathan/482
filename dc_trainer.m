function [U,S,V,threshold,w,sortim1,sortim2] = dc_trainer(im1,im2,feature)
    
    nd = size(im1,2);
    nc = size(im2,2);
    [U,S,V] = svd([im1 im2],'econ'); 
    projections = S*V';
    U = U(:,1:feature); % Add this in
    im1 = projections(1:feature,1:nd);
    im2 = projections(1:feature,nd+1:nd+nc);
    md = mean(im1,2);
    mc = mean(im2,2);

    Sw = 0;
    for k=1:nd
        Sw = Sw + (im1(:,k)-md)*(im1(:,k)-md)';
    end
    for k=1:nc
        Sw = Sw + (im2(:,k)-mc)*(im2(:,k)-mc)';
    end
    Sb = (md-mc)*(md-mc)';
    
    [V2,D] = eig(Sb,Sw);
    [lambda,ind] = max(abs(diag(D)));
    w = V2(:,ind);
    w = w/norm(w,2);
    vim1 = w'*im1;
    vim2 = w'*im2;
    
    if mean(vim1)>mean(vim2)
        w = -w;
        vim1 = -vim1;
        vim2 = -vim2;
    end
    
    % Don't need plotting here
    sortim1 = sort(vim1);
    sortim2 = sort(vim2);
    t1 = length(sortim1);
    t2 = 1;
    while sortim1(t1)>sortim2(t2)
    t1 = t1-1;
    t2 = t2+1;
    end
    threshold = (sortim1(t1)+sortim2(t2))/2;

    % We don't need to plot results
end


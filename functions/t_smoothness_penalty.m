function loss = t_smoothness_penalty(x,smoothness_l)

    loss = 0;

    for i=2:length(x)
        loss = loss + norm(x{i} - x{i-1},'fro')^2;
    end

    loss = smoothness_l * loss;

end
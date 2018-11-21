function [final_rippleLFP,final_rippleLocs] = inspectRipples(rippleSnips,rippleIdx)

%use this code to manually go through each SWR in the LFP and approve or
%delete the ripple
nRipples_start = length(rippleSnips);

t = 1;
for i = 1:nRipples_start
    if length(rippleSnips(i).lfp) < 2000
       lfpSnip = rippleSnips(i).lfp;
    else 
        lfpSnip = rippleSnips(i).lfp(500:2000);
    end
    figure; plot(lfpSnip);
    box off
    set(gca,'TickDir','out')
    xlim([1 length(lfpSnip)])
    
%display prompt and wait for response
    prompt = sprintf('Keep ripple? %d of %d',i,nRipples_start);
    x = input(prompt,'s');
    
    if strcmp(x,'y')
        keepRipples(t) = i;
        t = t + 1;
    end
    clear lfpSnip
    close
end

final_rippleLocs = rippleIdx(keepRipples);
final_rippleLFP = rippleSnips(keepRipples);
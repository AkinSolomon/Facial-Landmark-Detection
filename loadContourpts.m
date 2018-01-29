function [Vertices,Lines] = loadContourpts(p,pseudopts,verbose)


% Interpolate to get more points
r=5;
pointsx=interp(p.x,r); pointsx = pointsx(1:end-r+1);
pointsy=interp(p.y,r); pointsy = pointsy(1:end-r+1);

% Mark Landmark points with 1, other points zero
i=find(p.t==0); pointst=0; pointst((i-1)*r+1)=1;

if(verbose), imshow(p.I), hold on; %plot(pointsx,pointsy,'b'); 
end

% Find the Landmark point locations
i=find(pointst);

totalx=[]; totaly=[];
% Loop to make points evenly spaced on line pieces between landmark points

for j=1:length(i)-1
    % One line piece
    linex = pointsx(i(j):i(j+1));
    liney = pointsy(i(j):i(j+1));
    % Lenght on line through the points
    dx = [0 linex(2:end)-linex(1:end-1)];
    dy = [0 liney(2:end)-liney(1:end-1)];
    dist=cumsum(sqrt(dx.^2+dy.^2));
    % Interpolate new points evenly spaced on the line piece
    dist2=linspace(0,max(dist),pseudopts);
    linex=interp1(dist,linex,dist2);
    liney=interp1(dist,liney,dist2);
    
    % Display the line piece
    if(verbose)

        plot(liney,linex,'g*');
        plot(liney(1),linex(1),'r*');
        plot(liney(end),linex(end),'r*'); 
    end
    
   
  if(j<length(i)-1) 
      linex(end)=[];
      liney(end)=[];
  end  
    % Add the evenly spaced line piece to the total line
    totalx = [totalx linex];
    totaly = [totaly liney];
end
hold off
% Also store the image 
if(verbose)
    drawnow('expose'); 
end

Vertices = [totalx(:) totaly(:)];
Lines = [(1:size(Vertices,1))' ([2:size(Vertices,1) 1])'];
end
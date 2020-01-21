%% Sensor Degradation
figure
subplot(3,1,2)
fig = plot(tv,denorm("curvature",curvature),'--',tv,denorm("curvature",xh),'-',tv,denorm("curvature",xd),'-');
set(fig,'LineWidth',1)
axis([0,tv(end),-10,160])
xlabel('Time, $t$ (s)','Interpreter','latex')
ylabel('Resultant Curvature Angle, $x$ (deg)','Interpreter','latex')
%legend('Ground Truth','GRU-AUKF','DirectGRU','Location','best');
title('Curvature Estimation')
subplot(3,1,3)
fig = plot(tv,denorm("force",force),'--',tv,denorm("force",uh),'-',tv,denorm("force",ud),'-');
set(fig,'LineWidth',1)
axis([0,tv(end),-5,80])
xlabel('Time, $t$ (s)','Interpreter','latex')
ylabel('Tip Contact Force, $u_2$ (N)','Interpreter','latex')
legend('Ground Truth','GRU-AUKF','DirectGRU','Location','best');
title('Contact Force Estimation')
subplot(3,1,1)
fig = plot(tv,denorm("flex",sensor),tv,denorm("flex",flex));
set(fig,'LineWidth',1)
axis([0,tv(end),-0.5,1.5])
xlabel('Time, $t$ (s)','Interpreter','latex')
ylabel('Flex Sensor Reading, $y$ (V)','Interpreter','latex')
legend('Degraded','Original','Location','best');
title('Flex Sensor Measurement')
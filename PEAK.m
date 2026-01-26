signals = load("signals.mat")

voltage = signals.signals(:,1)
current = signals.signals(:,39)

for i = 20:length(voltage)
    plot(voltage, signals.signals(:, i))
    
end

%%
test = [2, 26, 62, 30, 32, 7, 59, 106, 68, 126, 167, 140, 130, 197, 147, 128, 105, 58, 88, 16, 15, 17, 2];
x = linspace(1, 100, length(test))
plot(x, test)
clear all;

SNR=0; % signal to noise ratio in dB;

N=20; % Number of bits 
Ns=50; % Number of time samples per bit.
test_square = 1;
% Setup either a square pulse or triangle pulse.
if test_square == 1
    t=([1:Ns]-0.5)/Ns; % Time axis for square pulse shape.
    p=ones(size(t)); % shape of the square pulse.
    % Set decision rule average cutoff
    cutoff = 25;
else
    t = 1:Ns; %define time from 1-50
    p = 1-abs(t/Ns); %triangle function with max height of 1.
    % Set decision rule average cutoff
    cutoff = 5;
end
d=fix(rand(1,N)+0.5); % data bits (random).

X(1:Ns:(Ns*(N-1)+1))=d;
X=conv(X,p); % PAM (Pulse Amplidute Modulation) signal with pulse shape.
sigma=1/(10^(SNR/20)); % Noise strength.
noise=sigma*randn(size(X)); % White Gaussian noise.
R=X+noise;

figure(1);
subplot(3,1,1);
plot(X);
title('Original Data');
subplot(3,1,2);
plot(R);
title('Noisy Data');

match_pulse = p(end:-1:1); % Just the opposite of the input pulse
Y = conv(match_pulse,R); % Convolve the filter with the signal
subplot(3,1,3);
Y(1001:1049) = []; % Remove the last 49 entries as they do not matter
plot(Y);
title('Output Data');

received_bits = size(d,2);
count = 1; % A counter for the for loop (i is doing other things)
for i=1:50:size(Y,2)
    % Calculate the slope every 50 elements
    slope=(Y(i+49)-Y(i))/50;
    % Really positive slope means 1
    if slope > 0.5
        received_bits(count) = 1;
    % Really negative slope means 0
    elseif slope < -0.5
        received_bits(count) = 0;
    % Almost no slope means same as previous value
    elseif (slope < 0.5) && (slope > -0.5)
        % First pass?
        if i == 1
            % Need to calculate the Average of the first 50 elements
            if mean(Y,[1,50]) > cutoff
                received_bits(count) = 1;
            else
                received_bits(count) = 0;
            end
        else
            received_bits(count) = received_bits(count-1);
        end
    end
    count = count + 1;
end

equality = isequal(d,received_bits);
ber = 0;
for i=1:size(d,2)
    if received_bits(i) ~= d(i)
        ber = ber + 1;
    end
end
ber = (ber/size(d,2))*100;

% PDF Estimate of the Gaussian White Noise at different SNR settings
rng default
figure(2);
h = histogram(noise, 'Normalization', 'probability');
title('White Noise PDF at SNR=0');

% Autocorrelation Estimate of the Gaussian White Noise 
% at a particular SNR setting.
lags = -999:1:999; % X-axis need 1999 elements to match xcorr
c = xcorr(noise); % Autocorrelation of Noise and Signal
c = c/(max(c));
figure(3);
stem(lags, c);
xlabel('Lag');
title('Autocorrelation of Noise');

% Autocorrelation Estimate of the Gaussian White Noise and the received
% signal corrupted by white noise at a particular SNR setting.
c = xcorr(R); % Autocorrelation of Noise and Signal
c = c/(max(c));
figure(4);
stem(lags, c);
xlabel('Lag');
title('Autocorrelation of Signal + Noise');
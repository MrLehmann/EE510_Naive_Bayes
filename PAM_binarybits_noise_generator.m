clear all;

SNR=10; % signal to noise ratio in dB;

N=20; % Number of bits 
Ns=50; % Number of time samples per bit.
t=([1:Ns]-0.5)/Ns; % Time axis for square pulse shape.
%p=ones(size(t)); % shape of the square pulse.
a = 2;
%t = -a:0.001:a; %define time from -2 to 2
p = 1-abs(t/a); %triangle function with max height of 1.
d=fix(rand(1,N)+0.5); % data bits (random).

X(1:Ns:(Ns*(N-1)+1))=d;
X=conv(X,p); % PAM (Pulse Amplidute Modulation) signal with pulse shape.
sigma=1/(10^(SNR/20)); % Noise strength.
noise=sigma*randn(size(X)); % White Gaussian noise.
R=X+noise;

figure(1);
subplot(3,1,1);
plot(X);
subplot(3,1,2);
plot(R);

%match_pulse = ones(size(t)); % Could have also just used p
match_pulse = p(end:-1:1); % Just the opposite of the input pulse
Y = conv(match_pulse,R); % Convolve the filter with the signal
subplot(3,1,3);
plot(Y);

Y(1001:1049) = []; % Remove the last 49 entries as they do not matter
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
            if mean(Y,[1,50]) > 25
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
%received_signal = mean(reshape(Y, 50, []));
%received_bits(received_signal < 25) = 0;
%received_bits(received_signal >= 25) = 1;
equality = isequal(d,received_bits)
ber = 0;
for i=1:size(d,2)
    if received_bits(i) ~= d(i)
        ber = ber + 1;
    end
end
(ber/size(d,2))*100


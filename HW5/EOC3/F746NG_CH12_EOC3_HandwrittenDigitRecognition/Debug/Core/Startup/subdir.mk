################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
S_SRCS += \
../Core/Startup/startup_stm32f746nghx.s 

S_DEPS += \
./Core/Startup/startup_stm32f746nghx.d 

OBJS += \
./Core/Startup/startup_stm32f746nghx.o 


# Each subdirectory must supply rules for building sources it contributes
Core/Startup/%.o: ../Core/Startup/%.s Core/Startup/subdir.mk
	arm-none-eabi-gcc -mcpu=cortex-m7 -g3 -DDEBUG -DARM_MATH_CM7 -DTF_LITE_STATIC_MEMORY -c -I"C:/Users/fusuy/Desktop/Embedded-Machine-Learning-with-Microcontrollers-Applications-on-STM32-Development-Boards-main/Chapter12/Application3-HDR/F746NG_CH12_EOC3_HandwrittenDigitRecognition/third_party/flatbuffers/include" -I"C:/Users/fusuy/Desktop/Embedded-Machine-Learning-with-Microcontrollers-Applications-on-STM32-Development-Boards-main/Chapter12/Application3-HDR/F746NG_CH12_EOC3_HandwrittenDigitRecognition/third_party/gemmlowp" -I"C:/Users/fusuy/Desktop/Embedded-Machine-Learning-with-Microcontrollers-Applications-on-STM32-Development-Boards-main/Chapter12/Application3-HDR/F746NG_CH12_EOC3_HandwrittenDigitRecognition/third_party/kissfft" -I"C:/Users/fusuy/Desktop/Embedded-Machine-Learning-with-Microcontrollers-Applications-on-STM32-Development-Boards-main/Chapter12/Application3-HDR/F746NG_CH12_EOC3_HandwrittenDigitRecognition/third_party/ruy" -I"C:/Users/fusuy/Desktop/Embedded-Machine-Learning-with-Microcontrollers-Applications-on-STM32-Development-Boards-main/Chapter12/Application3-HDR/F746NG_CH12_EOC3_HandwrittenDigitRecognition" -x assembler-with-cpp -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv5-sp-d16 -mfloat-abi=hard -mthumb -o "$@" "$<"

clean: clean-Core-2f-Startup

clean-Core-2f-Startup:
	-$(RM) ./Core/Startup/startup_stm32f746nghx.d ./Core/Startup/startup_stm32f746nghx.o

.PHONY: clean-Core-2f-Startup


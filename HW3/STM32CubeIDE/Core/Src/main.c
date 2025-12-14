/* USER CODE BEGIN Header */
/**
  ******************************************************************************
  * @file           : main.c
  * @brief          : Main program body - EE4065 HW3
  * @author         : Taner Kahyaoğlu
  * @date           : December 2025
  ******************************************************************************
  */
/* USER CODE END Header */

/* Includes ------------------------------------------------------------------*/
#include "main.h"

/* Private includes ----------------------------------------------------------*/
/* USER CODE BEGIN Includes */
#include "lib_image.h"
#include "lib_serialimage.h"
#include "string.h"
/* USER CODE END Includes */

/* Private typedef -----------------------------------------------------------*/
/* USER CODE BEGIN PTD */
/* USER CODE END PTD */

/* Private define ------------------------------------------------------------*/
/* USER CODE BEGIN PD */
#define IMG_WIDTH   128
#define IMG_HEIGHT  128
#define IMG_PIXELS  (IMG_WIDTH * IMG_HEIGHT)
/* USER CODE END PD */

/* Private macro -------------------------------------------------------------*/
/* USER CODE BEGIN PM */
/* USER CODE END PM */

/* Private variables ---------------------------------------------------------*/
UART_HandleTypeDef huart2;

/* USER CODE BEGIN PV */
/* =============================================================================
 * GLOBAL BUFFERS
 * =============================================================================
 */
uint32_t g_histogram_data[256];         // Histogram Array
uint8_t  g_processed_image[IMG_PIXELS]; // Destination Buffer for Morphology
uint8_t  g_tmp_image[IMG_PIXELS];       // Temp Buffer for Compound Operations

// Main Image Buffer (Double size to support RGB565)
volatile uint8_t pImage[IMG_PIXELS];

IMAGE_HandleTypeDef img;
/* USER CODE END PV */

/* Private function prototypes -----------------------------------------------*/
void SystemClock_Config(void);
static void MX_GPIO_Init(void);
static void MX_USART2_UART_Init(void);

/* USER CODE BEGIN PFP */
/* =============================================================================
 * HOMEWORK FUNCTION PROTOTYPES
 * =============================================================================
 */
// HW2 Helper
void Homework_Calculate_Histogram(uint8_t* p_gray, uint32_t* p_hist, uint32_t width, uint32_t height);

// HW3 - Q1/Q2: Otsu Thresholding
uint8_t Homework_Compute_Otsu_Threshold(uint32_t* p_hist, uint32_t total_pixels);
void Homework_Apply_Threshold(uint8_t* p_gray, uint32_t width, uint32_t height, uint8_t threshold);

// HW3 - Q3: Morphological Operations
void Homework_Dilation_3x3(uint8_t* p_src, uint8_t* p_dst, uint32_t width, uint32_t height);
void Homework_Erosion_3x3(uint8_t* p_src, uint8_t* p_dst, uint32_t width, uint32_t height);
void Homework_Opening_3x3(uint8_t* p_src, uint8_t* p_dst, uint32_t width, uint32_t height, uint8_t* p_tmp);
void Homework_Closing_3x3(uint8_t* p_src, uint8_t* p_dst, uint32_t width, uint32_t height, uint8_t* p_tmp);
/* USER CODE END PFP */

/* Private user code ---------------------------------------------------------*/
/* USER CODE BEGIN 0 */
/* USER CODE END 0 */

/**
  * @brief  The application entry point.
  * @retval int
  */
int main(void)
{
  /* USER CODE BEGIN 1 */
  /* USER CODE END 1 */

  /* MCU Configuration--------------------------------------------------------*/
  HAL_Init();

  /* USER CODE BEGIN Init */
  /* USER CODE END Init */

  /* Configure the system clock */
  SystemClock_Config();

  /* USER CODE BEGIN SysInit */
  /* USER CODE END SysInit */

  /* Initialize all configured peripherals */
  MX_GPIO_Init();
  MX_USART2_UART_Init();

  /* USER CODE BEGIN 2 */
  // Image structure initialization is handled inside the specific Question blocks
  // to support both Grayscale (Q1, Q3) and RGB565 (Q2).
  /* USER CODE END 2 */

  /* Infinite loop */
  /* USER CODE BEGIN WHILE */
  while (1)
  {
    /* USER CODE END WHILE */

    /* USER CODE BEGIN 3 */

    /* =========================================================================
     * QUESTION 1: OTSU THRESHOLDING (GRAYSCALE)
     * =========================================================================
     * - Receives Grayscale Image
     * - Calculates Single Otsu Threshold
     * - Binarizes the Image
     */
#if 0  // Set to 1 to ENABLE Q1, 0 to DISABLE
    img.format = IMAGE_FORMAT_GRAYSCALE;
    LIB_IMAGE_InitStruct(&img, (uint8_t*)pImage, IMG_HEIGHT, IMG_WIDTH, IMAGE_FORMAT_GRAYSCALE);

    if (LIB_SERIAL_IMG_Receive(&img) == SERIAL_OK)
    {
      // 1. Calculate Histogram
      Homework_Calculate_Histogram((uint8_t*)pImage, g_histogram_data, IMG_WIDTH, IMG_HEIGHT);

      // 2. Find Optimal Threshold
      uint8_t otsu_T = Homework_Compute_Otsu_Threshold(g_histogram_data, IMG_PIXELS);

      // 3. Apply Threshold
      Homework_Apply_Threshold((uint8_t*)pImage, IMG_WIDTH, IMG_HEIGHT, otsu_T);

      // 4. Send Result
      LIB_SERIAL_IMG_Transmit(&img);
    }
#endif

    /* =========================================================================
     * QUESTION 2: MULTI-CHANNEL OTSU (RGB565)
     * =========================================================================
     * - Receives RGB565 Image
     * - Calculates separate thresholds for R, G, B channels
     * - Reconstructs the image based on channel thresholds
     */
#if 0   // Set to 1 to ENABLE Q2, 0 to DISABLE
    img.format = IMAGE_FORMAT_RGB565;
    LIB_IMAGE_InitStruct(&img, (uint8_t*)pImage, IMG_HEIGHT, IMG_WIDTH, IMAGE_FORMAT_RGB565);

    if (LIB_SERIAL_IMG_Receive(&img) == SERIAL_OK)
    {
        uint16_t* p_src_rgb = (uint16_t*)pImage;

        // Static buffers to prevent stack overflow
        static uint32_t histR[256];
        static uint32_t histG[256];
        static uint32_t histB[256];

        // 1. Clear Histograms
        memset(histR, 0, 256 * sizeof(uint32_t));
        memset(histG, 0, 256 * sizeof(uint32_t));
        memset(histB, 0, 256 * sizeof(uint32_t));

        // 2. Build Histograms for R, G, B
        for (uint32_t i = 0; i < IMG_PIXELS; i++)
        {
            uint16_t pixel = p_src_rgb[i];
            uint8_t r = ((pixel >> 11) & 0x1F) << 3;
            uint8_t g = ((pixel >> 5)  & 0x3F) << 2;
            uint8_t b = ((pixel)       & 0x1F) << 3;

            histR[r]++; histG[g]++; histB[b]++;
        }

        // 3. Compute Thresholds per Channel
        uint8_t T_red   = Homework_Compute_Otsu_Threshold(histR, IMG_PIXELS);
        uint8_t T_green = Homework_Compute_Otsu_Threshold(histG, IMG_PIXELS);
        uint8_t T_blue  = Homework_Compute_Otsu_Threshold(histB, IMG_PIXELS);

        // 4. Apply Thresholds & Reconstruct
        for (uint32_t i = 0; i < IMG_PIXELS; i++)
        {
            uint16_t pixel = p_src_rgb[i];
            uint8_t r = ((pixel >> 11) & 0x1F) << 3;
            uint8_t g = ((pixel >> 5)  & 0x3F) << 2;
            uint8_t b = ((pixel)       & 0x1F) << 3;

            // Set to Max if > Threshold, else 0
            uint16_t new_r = (r > T_red)   ? 0x1F : 0;
            uint16_t new_g = (g > T_green) ? 0x3F : 0;
            uint16_t new_b = (b > T_blue)  ? 0x1F : 0;

            p_src_rgb[i] = (new_r << 11) | (new_g << 5) | (new_b);
        }

        LIB_SERIAL_IMG_Transmit(&img);
    }
#endif

    /* =========================================================================
     * QUESTION 3: MORPHOLOGICAL OPERATIONS
     * =========================================================================
     * - Receives Grayscale Image
     * - Auto-Binarizes using Otsu
     * - Applies selected Morphological Filter
     */
#if 1   // Set to 1 to ENABLE Q3, 0 to DISABLE
    img.format = IMAGE_FORMAT_GRAYSCALE;
    LIB_IMAGE_InitStruct(&img, (uint8_t*)pImage, IMG_HEIGHT, IMG_WIDTH, IMAGE_FORMAT_GRAYSCALE);

    if (LIB_SERIAL_IMG_Receive(&img) == SERIAL_OK)
    {
      // STEP 1: Pre-processing (Otsu Binarization)
      Homework_Calculate_Histogram((uint8_t*)pImage, g_histogram_data, IMG_WIDTH, IMG_HEIGHT);
      uint8_t otsu_T = Homework_Compute_Otsu_Threshold(g_histogram_data, IMG_PIXELS);
      Homework_Apply_Threshold((uint8_t*)pImage, IMG_WIDTH, IMG_HEIGHT, otsu_T);

      // STEP 2: Apply Morphology (Select ONE active block below)

#if 0 // --- Dilation ---
      Homework_Dilation_3x3((uint8_t*)pImage, g_processed_image, IMG_WIDTH, IMG_HEIGHT);
      memcpy((uint8_t*)pImage, g_processed_image, IMG_PIXELS);
#endif

#if 0 // --- Erosion ---
      Homework_Erosion_3x3((uint8_t*)pImage, g_processed_image, IMG_WIDTH, IMG_HEIGHT);
      memcpy((uint8_t*)pImage, g_processed_image, IMG_PIXELS);
#endif

#if 0 // --- Opening ---
      Homework_Opening_3x3((uint8_t*)pImage, g_processed_image, IMG_WIDTH, IMG_HEIGHT, g_tmp_image);
      memcpy((uint8_t*)pImage, g_processed_image, IMG_PIXELS);
#endif

#if 1 // --- Closing ---
      Homework_Closing_3x3((uint8_t*)pImage, g_processed_image, IMG_WIDTH, IMG_HEIGHT, g_tmp_image);
      memcpy((uint8_t*)pImage, g_processed_image, IMG_PIXELS);
#endif

      // STEP 3: Send Result
      LIB_SERIAL_IMG_Transmit(&img);
    }
#endif

    /* USER CODE END 3 */
  }
}

/**
  * @brief System Clock Configuration
  * @retval None
  */
void SystemClock_Config(void)
{
  RCC_OscInitTypeDef RCC_OscInitStruct = {0};
  RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

  /** Configure the main internal regulator output voltage */
  __HAL_RCC_PWR_CLK_ENABLE();
  __HAL_PWR_VOLTAGESCALING_CONFIG(PWR_REGULATOR_VOLTAGE_SCALE3);

  /** Initializes the RCC Oscillators */
  RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
  RCC_OscInitStruct.HSIState = RCC_HSI_ON;
  RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
  RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
  RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
  RCC_OscInitStruct.PLL.PLLM = 16;
  RCC_OscInitStruct.PLL.PLLN = 336;
  RCC_OscInitStruct.PLL.PLLP = RCC_PLLP_DIV4;
  RCC_OscInitStruct.PLL.PLLQ = 2;
  RCC_OscInitStruct.PLL.PLLR = 2;
  if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
  {
    Error_Handler();
  }

  /** Initializes the CPU, AHB and APB buses clocks */
  RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK|RCC_CLOCKTYPE_SYSCLK
                              |RCC_CLOCKTYPE_PCLK1|RCC_CLOCKTYPE_PCLK2;
  RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
  RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
  RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV2;
  RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;

  if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief USART2 Initialization Function
  * @param None
  * @retval None
  */
static void MX_USART2_UART_Init(void)
{
  huart2.Instance = USART2;
  huart2.Init.BaudRate   = 2000000;
  huart2.Init.WordLength = UART_WORDLENGTH_8B;
  huart2.Init.StopBits   = UART_STOPBITS_1;
  huart2.Init.Parity     = UART_PARITY_NONE;
  huart2.Init.Mode       = UART_MODE_TX_RX;
  huart2.Init.HwFlowCtl  = UART_HWCONTROL_NONE;
  huart2.Init.OverSampling = UART_OVERSAMPLING_16;
  if (HAL_UART_Init(&huart2) != HAL_OK)
  {
    Error_Handler();
  }
}

/**
  * @brief GPIO Initialization Function
  * @param None
  * @retval None
  */
static void MX_GPIO_Init(void)
{
  GPIO_InitTypeDef GPIO_InitStruct = {0};

  __HAL_RCC_GPIOC_CLK_ENABLE();
  __HAL_RCC_GPIOH_CLK_ENABLE();
  __HAL_RCC_GPIOA_CLK_ENABLE();
  __HAL_RCC_GPIOB_CLK_ENABLE();

  HAL_GPIO_WritePin(LD2_GPIO_Port, LD2_Pin, GPIO_PIN_RESET);

  GPIO_InitStruct.Pin = B1_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_IT_FALLING;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  HAL_GPIO_Init(B1_GPIO_Port, &GPIO_InitStruct);

  GPIO_InitStruct.Pin = LD2_Pin;
  GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
  GPIO_InitStruct.Pull = GPIO_NOPULL;
  GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
  HAL_GPIO_Init(LD2_GPIO_Port, &GPIO_InitStruct);
}

/* USER CODE BEGIN 4 */

/* =============================================================================
 * HW2 Helper: Calculate Histogram
 * =============================================================================
 */
void Homework_Calculate_Histogram(uint8_t* p_gray, uint32_t* p_hist,
                                  uint32_t width, uint32_t height)
{
  uint32_t i;
  uint32_t total_pixels = width * height;

  // 1. Initialize histogram array to 0
  for (i = 0; i < 256; i++)
    p_hist[i] = 0;

  // 2. Count pixel intensities
  for (i = 0; i < total_pixels; i++)
  {
    uint8_t pixel_value = p_gray[i];
    p_hist[pixel_value]++;
  }
}

/* =============================================================================
 * HW3: Compute Otsu Threshold
 * Calculates the optimal threshold that maximizes between-class variance.
 * =============================================================================
 */
uint8_t Homework_Compute_Otsu_Threshold(uint32_t* p_hist, uint32_t total_pixels)
{
  uint32_t sum_all = 0;
  uint32_t sumB = 0;
  uint32_t wB = 0;
  uint32_t wF = 0;
  float mB, mF;
  float var_between;
  float max_var_between = 0.0f;
  uint8_t threshold = 0;

  // Calculate total moment
  for (uint32_t i = 0; i < 256; i++)
    sum_all += i * p_hist[i];

  // Iterate through all possible thresholds
  for (uint32_t t = 0; t < 256; t++)
  {
    wB += p_hist[t];
    if (wB == 0) continue;

    wF = total_pixels - wB;
    if (wF == 0) break;

    sumB += t * p_hist[t];

    mB = (float)sumB / (float)wB;
    mF = (float)(sum_all - sumB) / (float)wF;

    // Calculate Between Class Variance
    float diff = mB - mF;
    var_between = (float)wB * (float)wF * diff * diff;

    if (var_between > max_var_between)
    {
      max_var_between = var_between;
      threshold = (uint8_t)t;
    }
  }
  return threshold;
}

/* =============================================================================
 * HW3: Apply Threshold (Binarization)
 * =============================================================================
 */
void Homework_Apply_Threshold(uint8_t* p_gray, uint32_t width,
                              uint32_t height, uint8_t threshold)
{
  uint32_t total_pixels = width * height;

  for (uint32_t i = 0; i < total_pixels; i++)
  {
    if (p_gray[i] > threshold)
      p_gray[i] = 255;
    else
      p_gray[i] = 0;
  }
}

/* =============================================================================
 * HW3: Dilation (Max Filter)
 * Set pixel to 255 if any neighbor is 255.
 * =============================================================================
 */
void Homework_Dilation_3x3(uint8_t* p_src, uint8_t* p_dst,
                           uint32_t width, uint32_t height)
{
  for (uint32_t y = 0; y < height; y++)
  {
    for (uint32_t x = 0; x < width; x++)
    {
      // Boundary Check: Ignore borders for simplicity (or padding logic)
      if (y == 0 || y == height - 1 || x == 0 || x == width - 1)
      {
        p_dst[y * width + x] = 0;
      }
      else
      {
        uint8_t max_val = 0;
        // Iterate 3x3 kernel
        for (int ky = -1; ky <= 1; ky++)
        {
          for (int kx = -1; kx <= 1; kx++)
          {
            uint8_t v = p_src[(y + ky) * width + (x + kx)];
            if (v > max_val) max_val = v;
          }
        }
        p_dst[y * width + x] = max_val;
      }
    }
  }
}

/* =============================================================================
 * HW3: Erosion (Min Filter)
 * Set pixel to 255 only if all neighbors are 255.
 * =============================================================================
 */
void Homework_Erosion_3x3(uint8_t* p_src, uint8_t* p_dst,
                          uint32_t width, uint32_t height)
{
  for (uint32_t y = 0; y < height; y++)
  {
    for (uint32_t x = 0; x < width; x++)
    {
      if (y == 0 || y == height - 1 || x == 0 || x == width - 1)
      {
        p_dst[y * width + x] = 0;
      }
      else
      {
        uint8_t min_val = 255;
        // Iterate 3x3 kernel
        for (int ky = -1; ky <= 1; ky++)
        {
          for (int kx = -1; kx <= 1; kx++)
          {
            uint8_t v = p_src[(y + ky) * width + (x + kx)];
            if (v < min_val) min_val = v;
          }
        }
        p_dst[y * width + x] = min_val;
      }
    }
  }
}

/* =============================================================================
 * HW3: Opening (Erosion followed by Dilation)
 * Removes small noise (salt).
 * =============================================================================
 */
void Homework_Opening_3x3(uint8_t* p_src, uint8_t* p_dst,
                          uint32_t width, uint32_t height,
                          uint8_t* p_tmp)
{
  Homework_Erosion_3x3(p_src, p_tmp, width, height);  // Step 1 -> Temp
  Homework_Dilation_3x3(p_tmp, p_dst, width, height); // Step 2 -> Dest
}

/* =============================================================================
 * HW3: Closing (Dilation followed by Erosion)
 * Fills small holes (pepper).
 * =============================================================================
 */
void Homework_Closing_3x3(uint8_t* p_src, uint8_t* p_dst,
                          uint32_t width, uint32_t height,
                          uint8_t* p_tmp)
{
  Homework_Dilation_3x3(p_src, p_tmp, width, height); // Step 1 -> Temp
  Homework_Erosion_3x3(p_tmp, p_dst, width, height);  // Step 2 -> Dest
}

/* USER CODE END 4 */

/**
  * @brief  This function is executed in case of error occurrence.
  * @retval None
  */
void Error_Handler(void)
{
  __disable_irq();
  while (1)
  {
  }
}

#ifdef USE_FULL_ASSERT
void assert_failed(uint8_t *file, uint32_t line)
{
  /* place for user assert code */
}
#endif /* USE_FULL_ASSERT */

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Loan Interest Rate Predictor',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        useMaterial3: true,
      ),
      home: const LoanPredictorScreen(),
    );
  }
}

class LoanPredictorScreen extends StatefulWidget {
  const LoanPredictorScreen({super.key});

  @override
  State<LoanPredictorScreen> createState() => _LoanPredictorScreenState();
}

// Dropdown option mappings
const Map<int, String> homeOwnershipMap = {
  0: 'Rent',
  1: 'Own',
  2: 'Mortgage',
  3: 'Other',
  4: 'Any',
  5: 'None',
};

const Map<int, String> loanPurposeMap = {
  0: 'Debt Consolidation',
  1: 'Credit Card',
  2: 'Home Improvement',
  3: 'Small Business',
  4: 'Personal Loan',
  5: 'Wedding',
  6: 'Medical',
  7: 'Car Financing',
  8: 'Moving',
  9: 'Vacation',
  10: 'Education',
  11: 'Energy Efficient',
  12: 'Other',
  13: 'Renewable Energy',
};

class _LoanPredictorScreenState extends State<LoanPredictorScreen> {
  // Form key for validation
  final GlobalKey<FormState> _formKey = GlobalKey<FormState>();

  // Controllers for numeric input fields
  final TextEditingController loanAmntController = TextEditingController();
  final TextEditingController annualIncController = TextEditingController();
  final TextEditingController dtiController = TextEditingController();

  // Dropdown field values
  int? selectedTerm;
  int? selectedSubGrade;
  int? selectedEmpLength;
  int? selectedHomeOwnership;
  int? selectedPurpose;

  bool isLoading = false;
  String? resultMessage;
  String? errorMessage;

  @override
  void dispose() {
    loanAmntController.dispose();
    annualIncController.dispose();
    dtiController.dispose();
    super.dispose();
  }

  Future<void> predictInterestRate() async {
    // Validate form first
    if (!_formKey.currentState!.validate()) {
      return;
    }

    // Clear previous results
    setState(() {
      resultMessage = null;
      errorMessage = null;
      isLoading = true;
    });

    try {
      // Create the request body with all 46 keys
      final Map<String, dynamic> requestBody = {
        // 8 primary fields from user input
        'loan_amnt': double.tryParse(loanAmntController.text) ?? 500,
        'term': selectedTerm ?? 36,
        'annual_inc': double.tryParse(annualIncController.text) ?? 1000,
        'sub_grade': selectedSubGrade ?? 1,
        'emp_length': selectedEmpLength ?? 0,
        'home_ownership': selectedHomeOwnership ?? 0,
        'purpose': selectedPurpose ?? 0,
        'dti': double.tryParse(dtiController.text) ?? 0,
        // 38 remaining fields with default values (meeting ge constraints)
        'installment': 10,
        'delinq_2yrs': 0,
        'inq_last_6mths': 0,
        'open_acc': 0,
        'pub_rec': 0,
        'revol_bal': 0,
        'revol_util': 0,
        'total_acc': 1,
        'collections_12_mths_ex_med': 0,
        'application_type': 0,
        'tot_coll_amt': 0,
        'tot_cur_bal': 0,
        'open_acc_6m': 0,
        'open_act_il': 0,
        'open_il_12m': 0,
        'open_il_24m': 0,
        'mths_since_rcnt_il': 0,
        'total_bal_il': 0,
        'il_util': 0,
        'open_rv_12m': 0,
        'open_rv_24m': 0,
        'max_bal_bc': 0,
        'all_util': 0,
        'total_rev_hi_lim': 0,
        'inq_fi': 0,
        'total_cu_tl': 0,
        'inq_last_12m': 0,
        'acc_open_past_24mths': 0,
        'avg_cur_bal': 0,
        'bc_open_to_buy': 0,
        'bc_util': 0,
        'mo_sin_old_il_acct': 0,
        'mo_sin_old_rev_tl_op': 0,
        'mo_sin_rcnt_rev_tl_op': 0,
        'mo_sin_rcnt_tl': 0,
        'mort_acc': 0,
        'mths_since_recent_bc': 0,
        'mths_since_recent_inq': 0,
        'verification_status': 0,
        'num_accts_ever_120_pd': 0,
        'num_actv_bc_tl': 0,
        'num_actv_rev_tl': 0,
        'num_bc_sats': 0,
        'num_bc_tl': 0,
        'num_il_tl': 0,
        'num_op_rev_tl': 0,
        'num_rev_accts': 0,
        'num_rev_tl_bal_gt_0': 0,
        'num_sats': 0,
        'num_tl_90g_dpd_24m': 0,
        'num_tl_op_past_12m': 0,
        'pct_tl_nvr_dlq': 0,
        'percent_bc_gt_75': 0,
        'pub_rec_bankruptcies': 0,
        'tot_hi_cred_lim': 0,
        'total_bal_ex_mort': 0,
        'total_bc_limit': 0,
        'total_il_high_credit_limit': 0,
      };

      // Make the POST request
      final response = await http.post(
        Uri.parse('https://linear-regression-model-2-42t2.onrender.com/predict'),
        headers: {
          'Content-Type': 'application/json',
        },
        body: json.encode(requestBody),
      );

      if (response.statusCode == 200) {
        // Parse the response
        final responseData = json.decode(response.body);
        final interestRate = responseData['predicted_interest_rate'] ?? 'N/A';
        
        setState(() {
          resultMessage = interestRate.toString();
          isLoading = false;
        });
      } else if (response.statusCode == 422) {
        // Validation error
        final errorData = json.decode(response.body);
        final errorDetail = errorData['detail'] ?? 'Validation error occurred';
        
        setState(() {
          errorMessage = 'Validation Error: $errorDetail';
          isLoading = false;
        });
      } else {
        // Other errors
        setState(() {
          errorMessage = 'Error: Server returned status ${response.statusCode}';
          isLoading = false;
        });
      }
    } catch (e) {
      setState(() {
        errorMessage = 'Error: ${e.toString()}';
        isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Loan Interest Rate Predictor'),
        centerTitle: true,
        elevation: 2,
      ),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Form(
            key: _formKey,
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                // Title section
                const SizedBox(height: 16),
                const Text(
                  'Enter Loan Details',
                  style: TextStyle(
                    fontSize: 24,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 8),
                const Text(
                  'All fields are required',
                  style: TextStyle(
                    fontSize: 14,
                    color: Colors.grey,
                  ),
                ),
                const SizedBox(height: 28),

                // Input Field 1: Loan Amount
                TextFormField(
                  controller: loanAmntController,
                  keyboardType: TextInputType.number,
                  decoration: InputDecoration(
                    labelText: 'Loan Amount (USD)',
                    hintText: 'e.g., 5000',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                    prefixIcon: const Icon(Icons.attach_money),
                  ),
                  validator: (value) {
                    if (value == null || value.isEmpty) {
                      return 'Loan Amount is required';
                    }
                    final amount = double.tryParse(value);
                    if (amount == null) {
                      return 'Please enter a valid number';
                    }
                    if (amount < 500 || amount > 40000) {
                      return 'Loan Amount must be between \$500 and \$40,000';
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 18),

                // Input Field 2: Term (Dropdown)
                DropdownButtonFormField<int>(
                  value: selectedTerm,
                  hint: const Text('Select Term'),
                  decoration: InputDecoration(
                    labelText: 'Loan Term (Months)',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                    prefixIcon: const Icon(Icons.calendar_today),
                  ),
                  items: const [
                    DropdownMenuItem(value: 36, child: Text('36 months')),
                    DropdownMenuItem(value: 60, child: Text('60 months')),
                  ],
                  onChanged: (value) {
                    setState(() => selectedTerm = value);
                  },
                  validator: (value) {
                    if (value == null) {
                      return 'Please select a term';
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 18),

                // Input Field 3: Annual Income
                TextFormField(
                  controller: annualIncController,
                  keyboardType: TextInputType.number,
                  decoration: InputDecoration(
                    labelText: 'Annual Income (USD)',
                    hintText: 'e.g., 60000',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                    prefixIcon: const Icon(Icons.trending_up),
                  ),
                  validator: (value) {
                    if (value == null || value.isEmpty) {
                      return 'Annual Income is required';
                    }
                    final income = double.tryParse(value);
                    if (income == null) {
                      return 'Please enter a valid number';
                    }
                    if (income < 1000 || income > 10000000) {
                      return 'Annual Income must be between \$1,000 and \$10,000,000';
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 18),

                // Input Field 4: Sub Grade (Dropdown)
                DropdownButtonFormField<int>(
                  value: selectedSubGrade,
                  hint: const Text('Select Sub Grade'),
                  decoration: InputDecoration(
                    labelText: 'Sub Grade',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                    prefixIcon: const Icon(Icons.grade),
                  ),
                  items: List.generate(
                    35,
                    (index) => DropdownMenuItem(
                      value: index + 1,
                      child: Text('Grade ${index + 1}'),
                    ),
                  ),
                  onChanged: (value) {
                    setState(() => selectedSubGrade = value);
                  },
                  validator: (value) {
                    if (value == null) {
                      return 'Please select a sub grade';
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 18),

                // Input Field 5: Employment Length (Dropdown)
                DropdownButtonFormField<int>(
                  value: selectedEmpLength,
                  hint: const Text('Select Employment Length'),
                  decoration: InputDecoration(
                    labelText: 'Employment Length (Years)',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                    prefixIcon: const Icon(Icons.work),
                  ),
                  items: List.generate(
                    11,
                    (index) => DropdownMenuItem(
                      value: index,
                      child: Text('$index years'),
                    ),
                  ),
                  onChanged: (value) {
                    setState(() => selectedEmpLength = value);
                  },
                  validator: (value) {
                    if (value == null) {
                      return 'Please select employment length';
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 18),

                // Input Field 6: Home Ownership (Dropdown)
                DropdownButtonFormField<int>(
                  value: selectedHomeOwnership,
                  hint: const Text('Select Home Ownership'),
                  decoration: InputDecoration(
                    labelText: 'Home Ownership',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                    prefixIcon: const Icon(Icons.home),
                  ),
                  items: homeOwnershipMap.entries
                      .map((e) => DropdownMenuItem(
                            value: e.key,
                            child: Text(e.value),
                          ))
                      .toList(),
                  onChanged: (value) {
                    setState(() => selectedHomeOwnership = value);
                  },
                  validator: (value) {
                    if (value == null) {
                      return 'Please select home ownership status';
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 18),

                // Input Field 7: Loan Purpose (Dropdown)
                DropdownButtonFormField<int>(
                  value: selectedPurpose,
                  hint: const Text('Select Loan Purpose'),
                  decoration: InputDecoration(
                    labelText: 'Loan Purpose',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                    prefixIcon: const Icon(Icons.info_outline),
                  ),
                  items: loanPurposeMap.entries
                      .map((e) => DropdownMenuItem(
                            value: e.key,
                            child: Text(e.value),
                          ))
                      .toList(),
                  onChanged: (value) {
                    setState(() => selectedPurpose = value);
                  },
                  validator: (value) {
                    if (value == null) {
                      return 'Please select a loan purpose';
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 18),

                // Input Field 8: DTI
                TextFormField(
                  controller: dtiController,
                  keyboardType: const TextInputType.numberWithOptions(
                    decimal: true,
                  ),
                  decoration: InputDecoration(
                    labelText: 'Debt-to-Income Ratio (DTI)',
                    hintText: 'e.g., 0.25',
                    border: OutlineInputBorder(
                      borderRadius: BorderRadius.circular(8),
                    ),
                    prefixIcon: const Icon(Icons.percent),
                  ),
                  validator: (value) {
                    if (value == null || value.isEmpty) {
                      return 'DTI is required';
                    }
                    final dti = double.tryParse(value);
                    if (dti == null) {
                      return 'Please enter a valid number';
                    }
                    if (dti < 0 || dti > 100) {
                      return 'DTI must be between 0 and 100';
                    }
                    return null;
                  },
                ),
                const SizedBox(height: 36),

                // Predict Button
                SizedBox(
                  width: double.infinity,
                  height: 54,
                  child: ElevatedButton(
                    onPressed: isLoading ? null : predictInterestRate,
                    style: ElevatedButton.styleFrom(
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(8),
                      ),
                    ),
                    child: isLoading
                        ? const SizedBox(
                            height: 24,
                            width: 24,
                            child: CircularProgressIndicator(
                              strokeWidth: 3,
                              valueColor: AlwaysStoppedAnimation<Color>(
                                Colors.white,
                              ),
                            ),
                          )
                        : const Text(
                            'Predict Interest Rate',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                  ),
                ),
                const SizedBox(height: 32),

                // Results Card
                if (resultMessage != null)
                  Card(
                    elevation: 4,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(24),
                      decoration: BoxDecoration(
                        color: Colors.green[50],
                        border: Border.all(
                          color: Colors.green,
                          width: 2,
                        ),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Column(
                        children: [
                          const Text(
                            'Predicted Interest Rate',
                            style: TextStyle(
                              fontSize: 14,
                              color: Colors.grey,
                              fontWeight: FontWeight.w500,
                            ),
                          ),
                          const SizedBox(height: 12),
                          Text(
                            '$resultMessage%',
                            textAlign: TextAlign.center,
                            style: TextStyle(
                              fontSize: 48,
                              fontWeight: FontWeight.bold,
                              color: Colors.green[700],
                            ),
                          ),
                          const SizedBox(height: 12),
                          Text(
                            'Annual Interest Rate',
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.grey[600],
                            ),
                          ),
                        ],
                      ),
                    ),
                  )
                else if (errorMessage != null)
                  Card(
                    elevation: 4,
                    shape: RoundedRectangleBorder(
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: Container(
                      width: double.infinity,
                      padding: const EdgeInsets.all(24),
                      decoration: BoxDecoration(
                        color: Colors.red[50],
                        border: Border.all(
                          color: Colors.red,
                          width: 2,
                        ),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Column(
                        children: [
                          Icon(
                            Icons.error_outline,
                            color: Colors.red[700],
                            size: 40,
                          ),
                          const SizedBox(height: 12),
                          Text(
                            'Error',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.bold,
                              color: Colors.red[700],
                            ),
                          ),
                          const SizedBox(height: 8),
                          Text(
                            errorMessage!,
                            textAlign: TextAlign.center,
                            style: TextStyle(
                              color: Colors.red[700],
                              fontSize: 14,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ),
                const SizedBox(height: 24),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

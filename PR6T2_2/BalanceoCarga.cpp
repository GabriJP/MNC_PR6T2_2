void BalanceoCarga(int Nth, int M, int *Pos, int *Num){
	if (M <= Nth){
		for (int i = 0; i < Nth; i++) Pos[i] = i;
		for (int i = 0; i < M; i++) Num[i] = 1;
		for (int i = M; i < Nth; i++) Num[i] = 0;
		return;
	}
	int a = M / Nth;
	for (int i = 0; i < Nth; i++) Num[i] = a;
	int b = M - a*Nth;
	if (b>0){
		for (int i = 0; i < b; i++) Num[i]++;
	}
	Pos[0] = 0;
	for (int i = 1; i < Nth; i++) Pos[i] = Pos[i - 1] + Num[i - 1];
}